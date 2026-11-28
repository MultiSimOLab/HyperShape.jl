# Augmented Lagrangian: an oracle that wraps another one, keeps the constraints the optimiser treats itself (native)
# and moves the others into the objective, with multipliers updated after every accepted iteration.

"""
    MultiplierUpdate

Supertype of the rules of an augmented Lagrangian ([`Bertsekas`](@ref)). A rule implements (all public in
`HyperShape.Optimizers`, not exported):

- `term(u, g, λ, μ, c::ConstraintInfo)`: what the constraint adds to the objective;
- `weight(u, g, λ, μ, c)`: `∂term/∂g`;
- `initial_penalty(u)`: the penalty `μ` of every absorbed constraint at the start;
- `update!(u, al::AugmentedLagrangianOracle)`: new `al.λ[i]` and `al.μ[i]` for `i in al.absorbed` from the values
  `al.g[i]` at the accepted design (`al.infos[i]` their bounds, `al.updates` the number of updates including this one,
  counted by `end_iteration!`, `al.vref` free for the rule); returns whether anything changed.
"""
abstract type MultiplierUpdate end

"""
    Bertsekas(; μ0 = 1.0, growth = 1.0, progress = 0.25, window = 50, μmax = 100.0)

Classical multiplier update (Bertsekas; Rockafellar's form for inequalities and intervals). With `y = g + λ/μ` and
`Π` the projection onto the bounds `[lo, hi]` of the constraint:

    term = μ/2 (y − Π(y))² − λ²/(2μ),    weight = μ (y − Π(y)),    λ ← weight

that is `λ ← max(0, λ + μ g)` for `g ≤ 0` and `λ ← λ + μ g` for an equality (`λ > 0` when the upper bound is active,
`λ < 0` when the lower one is). The penalty `μ` starts at `μ0`. With `growth > 1` it is multiplied by `growth` (up
to `μmax`) every `window` iterations in which the violation is above the tolerance of the augmented Lagrangian and did
not fall below `progress` times the one of the last check (`window` iterations before; the first check compares with
the first iteration); `progress = Inf` grows it at every check while the constraint is violated. `growth = 1`
(default) keeps it fixed (then `progress` and `window` have no effect: giving them is an error; `μmax` is not used
and any `μ0` is accepted), which is what
first-order optimisers such as SiMPL need: a large penalty makes the subproblem ill-conditioned. The scaled
functionals are of order one, so `μ0 = 1` is the neutral value. `Bertsekas(μ0 = 0.2, growth = 1.025, window = 1,
progress = Inf, μmax = 2.0)` is the rule of Kim et al. (2020).
"""
struct Bertsekas <: MultiplierUpdate
  μ0::Float64
  growth::Float64
  progress::Float64
  window::Int
  μmax::Float64
  function Bertsekas(; μ0::Real = 1.0, growth::Real = 1.0, progress::Union{Nothing,Real} = nothing,
                     window::Union{Nothing,Integer} = nothing, μmax::Union{Nothing,Real} = nothing)
    if growth == 1
      # μmax is accepted (as in earlier versions, Bertsekas(μ0 = 200.0, μmax = 300.0)) but neither checked nor used
      given = [string(k) for (k, v) in ((:progress, progress), (:window, window)) if v !== nothing]
      isempty(given) || throw(ArgumentError("Bertsekas: $(join(given, ", ")) $(length(given) == 1 ? "has" : "have") no " *
                                            "effect with growth = 1 (a fixed penalty): leave $(length(given) == 1 ? "it" : "them") out, or give growth > 1"))
    end
    progress, window, μmax = something(progress, 0.25), something(window, 50), something(μmax, 100.0)
    (isfinite(μ0) && μ0 > 0) || throw(ArgumentError("Bertsekas: μ0 must be positive, got $μ0"))
    growth >= 1 || throw(ArgumentError("Bertsekas: growth must be at least 1, got $growth"))
    progress > 0 || throw(ArgumentError("Bertsekas: progress must be positive, got $progress"))
    window >= 1 || throw(ArgumentError("Bertsekas: window must be at least 1, got $window"))
    growth == 1 || μmax >= μ0 || throw(ArgumentError("Bertsekas: μmax must be at least μ0"))
    new(Float64(μ0), Float64(growth), Float64(progress), Int(window), Float64(μmax))
  end
end

function Base.show(io::IO, u::Bertsekas)
  print(io, "Bertsekas(μ0 = ", u.μ0, ", growth = ", u.growth)
  u.growth > 1 && print(io, ", progress = ", u.progress, ", window = ", u.window, ", μmax = ", u.μmax)
  print(io, ")")
end

# equal vectors, by a reduction (no scalar indexing)
_same(x, y) = x === y || mapreduce(==, &, x, y; init = true)
_same(x::Vector, y::Vector) = x == y

@inline _excess(y, c::ConstraintInfo) = y - clamp(y, c.lo, c.hi)
term(::Bertsekas, g, λ, μ, c::ConstraintInfo) = (d = _excess(g + λ / μ, c); μ / 2 * d^2 - λ^2 / (2μ))
weight(::Bertsekas, g, λ, μ, c::ConstraintInfo) = μ * _excess(g + λ / μ, c)
initial_penalty(u::Bertsekas) = u.μ0

function update!(u::Bertsekas, al)
  changed = false
  check = u.growth > 1 && al.updates % u.window == 0
  for i in al.absorbed
    c = al.infos[i]
    v = violation(c, al.g[i])
    al.updates == 1 && (al.vref[i] = v)          # the reference of the first check
    λn = weight(u, al.g[i], al.λ[i], al.μ[i], c)
    changed |= λn != al.λ[i]
    al.λ[i] = λn
    if check
      if v > al.tol && (isinf(u.progress) || v > u.progress * al.vref[i]) && al.μ[i] < u.μmax
        al.μ[i] = min(u.growth * al.μ[i], u.μmax)
        changed = true
      end
      al.vref[i] = v
    end
  end
  changed
end

"""
    AugmentedLagrangian(update = Bertsekas(); native = :auto, constraint_tol = 1e-4)

How a method ([`DensityMethod`](@ref HyperShape.Methods.DensityMethod),
[`ReactionDiffusion`](@ref HyperShape.Methods.ReactionDiffusion)) treats the constraints its step does not treat
itself (see [`handles`](@ref)): they enter the objective of the step through the rule `update`
([`Bertsekas`](@ref)), with multipliers updated after every accepted iteration.

- `native = :auto`: in the order of the problem, the first constraints the optimiser handles (up to
  [`max_native`](@ref)) stay with it, the rest go to the augmented Lagrangian (`ReactionDiffusion`: the constraint
  built with `Volume`; none with `volume = nothing`); or a tuple of names, `(:volume,)`, or `()` (all to the
  augmented Lagrangian; OC needs one constraint of its own). It is the one place where the constraints of the step
  are named.
- `constraint_tol`: the run converges only when every constraint of the augmented Lagrangian is met to it (scaled
  value).

A run resumed from a [`Checkpoint`](@ref HyperShape.Methods.Checkpoint) continues the multipliers and penalties saved
there (the `μ0` given now is not applied to them).

Used by default when the problem has constraints the optimiser does not handle; the banner says so.
"""
struct AugmentedLagrangian{U<:MultiplierUpdate,N}
  update::U
  native::N
  constraint_tol::Float64
  function AugmentedLagrangian(update::U = Bertsekas(); native = :auto, constraint_tol::Real = 1e-4) where {U<:MultiplierUpdate}
    native === :auto || native isa Tuple{Vararg{Symbol}} ||
      throw(ArgumentError("AugmentedLagrangian: native must be :auto or a tuple of names, (:volume,) or ()"))
    constraint_tol > 0 || throw(ArgumentError("AugmentedLagrangian: constraint_tol must be positive"))
    new{U,typeof(native)}(update, native, Float64(constraint_tol))
  end
end

Base.show(io::IO, a::AugmentedLagrangian) =
  print(io, "AugmentedLagrangian(", a.update, "; native = ", repr(a.native), ", constraint_tol = ", a.constraint_tol, ")")

"""
    AugmentedLagrangianOracle(inner, al::AugmentedLagrangian, native, z)

Oracle of the augmented problem: the objective is `f + Σ term(gᵢ)` over the absorbed constraints (all but the
indices `native` of `inner`), whose gradient is one `weighted_gradient!` of `inner`; the native constraints are
passed through, renumbered `1, …, length(native)`; `z` gives the vector type. Like any oracle, `gradient!` is only
valid at the `z` of the last `objective` (it throws otherwise). The multipliers `λ` and penalties `μ` are vectors over all the
constraints of `inner` (`NaN` in the native ones). [`end_iteration!`](@ref) updates them at an accepted design.
"""
mutable struct AugmentedLagrangianOracle{O<:AbstractOracle,U<:MultiplierUpdate,V} <: AbstractOracle
  inner::O
  update::U
  native::Vector{Int}
  absorbed::Vector{Int}
  infos::Vector{ConstraintInfo}
  λ::Vector{Float64}
  μ::Vector{Float64}
  g::Vector{Float64}          # values of the absorbed constraints at the last objective
  vref::Vector{Float64}       # violation at the last check of the penalty
  w::Vector{Float64}          # weights of weighted_gradient!
  tol::Float64
  updates::Int
  version::Int                # grows when the multipliers or penalties change
  f::Float64                  # objective of the problem at the last objective call
  snap_z::V
  snap_dL::V
  snap_version::Int
  snap_valid::Bool
  zlast::V                    # the design of the last objective
end

function AugmentedLagrangianOracle(inner::AbstractOracle, al::AugmentedLagrangian, native, z)
  m = num_constraints(inner)
  native = collect(Int, native)
  all(i -> 1 <= i <= m, native) || throw(ArgumentError("AugmentedLagrangianOracle: native index out of range"))
  allunique(native) || throw(ArgumentError("AugmentedLagrangianOracle: a native constraint is repeated"))
  absorbed = [i for i in 1:m if !(i in native)]
  infos = [constraint_info(inner, i) for i in 1:m]
  λ = [i in native ? NaN : 0.0 for i in 1:m]
  μ = [i in native ? NaN : initial_penalty(al.update) for i in 1:m]
  AugmentedLagrangianOracle(inner, al.update, native, absorbed, infos, λ, μ, fill(NaN, m), fill(Inf, m), zeros(m),
                            al.constraint_tol, 0, 0, NaN, similar(z), similar(z), -1, false, fill!(similar(z), NaN))
end

function objective(al::AugmentedLagrangianOracle, z)
  copyto!(al.zlast, z)
  f = objective(al.inner, z)
  al.f = f
  isfinite(f) || return f
  L = f
  for i in al.absorbed
    g = constraint(al.inner, i, z)
    al.g[i] = g
    isfinite(g) || return g
    L += term(al.update, g, al.λ[i], al.μ[i], al.infos[i])
  end
  L
end

function gradient!(dL, al::AugmentedLagrangianOracle, z)
  if al.snap_valid && al.snap_version == al.version && _same(z, al.snap_z)
    return copyto!(dL, al.snap_dL)
  end
  _same(z, al.zlast) || throw(ArgumentError("AugmentedLagrangianOracle: gradient! asked at a design different from the last objective"))
  fill!(al.w, 0.0)
  for i in al.absorbed
    al.w[i] = weight(al.update, al.g[i], al.λ[i], al.μ[i], al.infos[i])
  end
  weighted_gradient!(dL, al.inner, 1.0, al.w, z)
  copyto!(al.snap_z, z)
  copyto!(al.snap_dL, dL)
  al.snap_version = al.version
  al.snap_valid = true
  dL
end

bounds(al::AugmentedLagrangianOracle) = bounds(al.inner)
num_constraints(al::AugmentedLagrangianOracle) = length(al.native)
constraint_info(al::AugmentedLagrangianOracle, j) = al.infos[al.native[j]]
linear_constraint(al::AugmentedLagrangianOracle, j) = linear_constraint(al.inner, al.native[j])
constraint(al::AugmentedLagrangianOracle, j::Integer, z) = constraint(al.inner, al.native[j], z)
constraint_gradient!(dG, al::AugmentedLagrangianOracle, j::Integer, z) = constraint_gradient!(dG, al.inner, al.native[j], z)
inner(al::AugmentedLagrangianOracle, x, y) = inner(al.inner, x, y)
riesz!(g, al::AugmentedLagrangianOracle, dF) = riesz!(g, al.inner, dF)

function weighted_gradient!(dL, al::AugmentedLagrangianOracle, w0, w, z)
  fill!(al.w, 0.0)
  for i in al.absorbed
    al.w[i] = w0 * weight(al.update, al.g[i], al.λ[i], al.μ[i], al.infos[i])
  end
  for (j, i) in enumerate(al.native)
    al.w[i] = w[j]
  end
  weighted_gradient!(dL, al.inner, w0, al.w, z)
end

"""
    end_iteration!(oracle, z) -> Bool

Called by the optimisation loop after every accepted iteration, at the accepted design `z`. An oracle whose objective
depends on the iteration (the augmented Lagrangian: its multipliers) updates itself and returns `true` when its
objective changed; the loop then calls [`objective_changed!`](@ref) on the optimiser. Default: nothing, `false`.
"""
end_iteration!(::AbstractOracle, z) = false

function end_iteration!(al::AugmentedLagrangianOracle, z)
  # values at the accepted design (cached: no solve; the last evaluation may have been a rejected trial)
  for i in al.absorbed
    al.g[i] = constraint(al.inner, i, z)        # the last evaluation may have been a rejected trial
  end
  al.updates += 1
  changed = update!(al.update, al)
  changed && (al.version += 1)
  changed
end

"""
    multipliers(al::AugmentedLagrangianOracle), penalties(al)

Multipliers `λ` and penalties `μ`, one per constraint of the inner oracle (`NaN` in the native ones).
"""
multipliers(al::AugmentedLagrangianOracle) = al.λ
penalties(al::AugmentedLagrangianOracle) = al.μ

"""
    objective_changed!(opt::Optimizer, state) -> state

Tells the optimiser that its objective changed between two iterations (the multipliers of an augmented Lagrangian)
while the design and the bounds did not. Default: [`reset!`](@ref). SiMPL forgets its value and gradient but keeps
its step size (Barzilai–Borwein), without which an augmented Lagrangian updated every iteration converges much more
slowly; OC keeps nothing between steps.
"""
objective_changed!(opt::Optimizer, state) = reset!(opt, state)
objective_changed!(::OC, s) = s
function objective_changed!(::SiMPL, s::SiMPLState)
  s.have_f = false
  s.have_grad = false
  s.can_rollback = false      # the value saved for a rollback belongs to the previous objective
  s
end
