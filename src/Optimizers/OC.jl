"""
    OC(; move = 0.2, damping = 0.5, bisection_tol = 1e-3, max_bisections = 200, constraint_tol = 1e-4,
         max_corrections = 8, adapt = (0.7, 1.2), move_min = 1e-3)

Optimality criteria update of Andreassen et al. (top88, 2011) for one constraint `g(z) ≤ 0`:

    zᵢ ← clamp(zᵢ Bᵢ^damping, max(lbᵢ, zᵢ - m), min(ubᵢ, zᵢ + m)),   Bᵢ = max(0, -∂F/∂zᵢ) / (λ aᵢ),

with `m = move * (ub - lb)` and `λ > 0` found by bisection on `aᵀz = V`, starting from `[0, 1e9]`.

- **Linear constraint** `aᵀz ≤ V`: the bisection stops when `(λ₂ - λ₁)/(λ₁ + λ₂) ≤ bisection_tol` and, as in top88,
  keeps the design of the last midpoint, so the constraint holds up to the bisection tolerance. With
  `adapt = nothing` the step is the one of top88. Requires `a > 0`.
- **Nonlinear constraint** (for example the volume of a projected density): `a = ∂g/∂z` at the current design (floored
  at `1e-12 maximum(a)`, since a filter can leave tiny or negative entries where a projection saturates; an entry
  with a zero derivative moves up to the move limit in the direction of descent) and the
  linearised constraint `aᵀz ≤ aᵀzₖ - g(zₖ) - s`. The shift `s` is then corrected (chord, then bounded secant steps,
  at most `max_corrections`) until `|g(z_new)| ≤ constraint_tol`, evaluating the true `g` once per correction; `s` is
  kept for the next step (`reset!` clears it). The bisection is solved to `min(bisection_tol, 1e-7)` so that the
  corrections see a smooth function. If the move limit does not allow meeting the constraint, the step is still taken.

**Move limits.** With `adapt = (dec, inc)` (default) each entry has its own move limit `mᵢ`: multiplied by `dec`
when its last step changed sign with respect to the one before, by `inc` when it kept it, and kept in
`[move_min, move]` (as the move limits of MMA, Svanberg 1987). It damps the oscillation of OC at the interface when a
tanh projection is sharp (a step of `move` is wider than the projection band, about `1/β`). `reset!` (a
continuation jump) restores every `mᵢ` to `move`; `shrink!` scales them all, and the retry after a rejected step does
not update them. `adapt = nothing` gives the scalar move
limit of top88 (then `move_min` has no effect: giving it is an error). `move` can follow a continuation, `Continuation(β = proj => […], move = oc => […])` (see
[`set_move!`](@ref)); with `adapt` it is the ceiling of the limits. The `step` of the [`StepInfo`](@ref) is the mean
of the limits over all entries (those at a bound included).

The objective is evaluated once per step and the new design is not evaluated (`f_new = NaN`, `kkt = NaN`). If the
move limit does not allow meeting the constraint (infeasible start), the step is still taken and `violation > 0` in
the [`StepInfo`](@ref).

It also reports the stationarity measure of the design before the step ([`stationarity!`](@ref), with the metric of
the oracle, which the step does not use).
"""
mutable struct OC <: Optimizer
  move::Float64
  damping::Float64
  bisection_tol::Float64
  max_bisections::Int
  constraint_tol::Float64
  max_corrections::Int
  adapt::Union{Nothing,Tuple{Float64,Float64}}
  move_min::Float64
end

function OC(; move = 0.2, damping = 0.5, bisection_tol = 1e-3, max_bisections = 200, constraint_tol = 1e-4,
            max_corrections = 8, adapt = (0.7, 1.2), move_min = nothing)
  adapt === nothing && move_min !== nothing &&
    throw(ArgumentError("OC: move_min has no effect with adapt = nothing (the scalar move limit of top88): leave it out"))
  move_min = something(move_min, 1e-3)
  0 < move || throw(ArgumentError("move must be positive"))
  0 < damping || throw(ArgumentError("damping must be positive"))
  0 < constraint_tol || throw(ArgumentError("constraint_tol must be positive"))
  max_corrections >= 0 || throw(ArgumentError("max_corrections must be non-negative"))
  if adapt !== nothing
    (length(adapt) == 2 && 0 < adapt[1] < 1 <= adapt[2]) ||
      throw(ArgumentError("adapt must be (decrease, increase) with 0 < decrease < 1 ≤ increase, or nothing"))
    0 < move_min <= move || throw(ArgumentError("move_min must be in (0, move]"))
  end
  OC(move, damping, bisection_tol, max_bisections, constraint_tol, max_corrections,
     adapt === nothing ? nothing : (Float64(adapt[1]), Float64(adapt[2])), Float64(move_min))
end

Base.setproperty!(::OC, ::Symbol, v) = throw(ArgumentError("the fields of OC cannot be set; change the move limit with set_move!"))

_valid_move(opt::OC, move) = isfinite(move) && move > 0 && (opt.adapt === nothing || move >= opt.move_min)

"""
    set_move!(opt::OC, move) -> opt

Changes the move limit of `opt` (the ceiling of the adaptive limits; with `adapt` it must be at least `move_min`).
Used by a `Continuation` of `move`; the optimisation loop calls `reset!` after it, which restores the adaptive limits
to the new value. Without `reset!` the limits are capped at the new value from the next step.
"""
function set_move!(opt::OC, move)
  _valid_move(opt, move) || throw(ArgumentError("move must be positive" * (opt.adapt === nothing ? "" : " and at least move_min = $(opt.move_min)")))
  setfield!(opt, :move, Float64(move))
  opt
end

mutable struct OCState{V}
  move::Float64  # current ceiling (opt.move, times the factors of shrink!)
  shrunk::Bool   # shrink! was called since the last step!
  dF::V
  z_old::V
  a::V           # derivative of a nonlinear constraint at the current design
  shift::Float64 # correction of the linearised constraint, kept between steps
  m::V           # adaptive move limit of each entry (with adapt)
  meff::V        # m scaled by the factors of shrink!
  dprev::V       # previous step, for the sign test
  k::Int         # steps taken since init or reset!
  sg::V          # stationarity measure: M⁻¹ ∂F/∂z, M⁻¹a and a work vector
  sga::V
  sw::V
end

function init(opt::OC, z, oracle)
  num_constraints(oracle) == 1 ||
    throw(ArgumentError("OC needs exactly one constraint (aᵀz ≤ V or g(z) ≤ 0); the oracle has $(num_constraints(oracle))"))
  handles(opt, constraint_info(oracle, 1)) ||
    throw(ArgumentError("OC treats one constraint g(z) ≤ 0 that solves no state; the oracle gives $(constraint_info(oracle, 1))"))
  c = linear_constraint(oracle, 1)
  c === nothing || minimum(c[1]) > 0 || throw(ArgumentError("OC needs a linear constraint with a > 0"))
  v() = similar(z)
  s = OCState(opt.move, false, v(), copy(z), v(), 0.0, v(), v(), v(), 0, v(), v(), v())
  fill!(s.m, opt.move)
  fill!(s.dprev, 0)
  s
end

function reset!(opt::OC, s::OCState)
  s.move = opt.move
  s.shrunk = false
  s.shift = 0.0
  fill!(s.m, opt.move)
  fill!(s.dprev, 0)
  s.k = 0
  s
end
# memory for a checkpoint: adaptive limits, last steps and correction shift
memory(::OC, s::OCState, z) = (started = Float64(s.k > 0), shift = s.shift, m = s.m, dprev = s.dprev, z_old = s.z_old)

function restore_memory!(::OC, s::OCState, z, v::NamedTuple)
  _check_memory(v, (:started, :shift, :m, :dprev, :z_old), length(z), "OC")
  s.k = v.started > 0 ? 1 : 0
  s.shift = v.shift
  copyto!(s.m, v.m)
  copyto!(s.dprev, v.dprev)
  copyto!(s.z_old, v.z_old)
  s
end

shrink!(opt::OC, s::OCState, factor) = (s.move *= factor; s.shrunk = true; s)
constraint_tol(opt::OC) = opt.constraint_tol

# Adaptive move limits (with adapt = (dec, inc)): each entry's limit is multiplied by dec when its last step changed
# sign with respect to the one before, by inc when it kept it, and clamped to [move_min, move]. Returns the move
# limit to use: the vector of limits scaled by the shrink! factors, or the scalar s.move without adapt.
# A retry after a rejected step (`retry`) only rescales: the step was rolled back, so there is no new sign to test.
function _move_limits!(opt::OC, s::OCState, z, retry::Bool = false)
  adapt = opt.adapt
  adapt === nothing && return s.move
  _adapt_limits!(opt, s, z, adapt[1], adapt[2], retry)
end

function _adapt_limits!(opt::OC, s::OCState, z, dec::Float64, inc::Float64, retry::Bool)
  if s.k > 0 && !retry
    lo, hi = opt.move_min, opt.move
    # z - z_old is the last step, s.dprev the one before
    @. s.m = clamp(s.m * ifelse((z - s.z_old) * s.dprev < 0, dec, ifelse((z - s.z_old) * s.dprev > 0, inc, 1.0)), lo, hi)
    @. s.dprev = z - s.z_old
  end
  sc = s.move / opt.move
  hi = opt.move
  @. s.meff = min(s.m, hi) * sc
  s.meff
end

_step_size(m::Real) = m
_step_size(m::AbstractVector) = sum(m) / length(m)

_oc_power(x, η) = η == 0.5 ? sqrt(x) : x^η

# B holds (max(0, -∂F/∂z) / a)^η and κ = λ^(-η), so the power is not recomputed at every bisection
function _oc_update!(z, zo, B, a, lb, ub, κ, m)
  @. z = clamp(zo * B * κ, max(lb, zo - m * (ub - lb)), min(ub, zo + m * (ub - lb)))
  dot(a, z)
end

# Bisection of λ on aᵀz = V; z keeps the update of the last midpoint (top88). Returns (aᵀz, λ, reached, active):
# `reached` is false when even λ = 1e69 leaves aᵀz > V (the move limit does not allow meeting the target), `active`
# is false when no λ tried left aᵀz > V (the constraint is inactive: λ → 0)
function _oc_bisect!(z, zo, B, a, lb, ub, V, m, η, tol, max_bisections)
  λ1, λ2 = 0.0, 1e9
  n = 0
  vol = _oc_update!(z, zo, B, a, lb, ub, λ2^(-η), m)
  while vol > V && n < 60   # widen the bracket if the objective is badly scaled
    λ1, λ2 = λ2, 10λ2
    vol = _oc_update!(z, zo, B, a, lb, ub, λ2^(-η), m)
    n += 1
  end
  reached = vol <= V
  n = 0
  λ = λ2
  while (λ2 - λ1) / (λ1 + λ2) > tol && n < max_bisections
    λ = (λ1 + λ2) / 2
    vol = _oc_update!(z, zo, B, a, lb, ub, λ^(-η), m)   # as in top88, z keeps the update of the last midpoint
    vol > V ? (λ1 = λ) : (λ2 = λ)
    n += 1
  end
  vol, λ, reached, λ1 > 0
end

function step!(opt::OC, s::OCState, z, oracle, ctx)
  retry = s.shrunk
  retry || (s.move = opt.move)   # the previous step stood: the reduction is dropped
  s.shrunk = false
  # function barrier: the move limit is a scalar (top88) or a vector (adaptive), each step specialised on it
  m = _move_limits!(opt, s, z, retry)
  s.k += 1
  m isa Float64 ? _oc_step!(opt, s, z, oracle, m) : _oc_step!(opt, s, z, oracle, m::typeof(s.meff))
end

function _oc_step!(opt::OC, s::OCState, z, oracle, m)
  f = objective(oracle, z)
  isfinite(f) || return StepInfo(false, f, NaN, NaN, 0.0, _step_size(m), 0.0, 1, NaN, NaN)
  gradient!(s.dF, oracle, z)
  c = linear_constraint(oracle, 1)
  c === nothing && return _oc_step_nonlinear!(opt, s, z, oracle, f, m)
  a, V = c
  lb, ub = bounds(oracle)
  stat = _oc_stationarity!(s, oracle, z, a, V)
  copyto!(s.z_old, z)
  zo, B, η = s.z_old, s.dF, opt.damping
  @. B = _oc_power(max(zero(B), -B) / a, η)
  vol, λ, _, _ = _oc_bisect!(z, zo, B, a, lb, ub, V, m, η, opt.bisection_tol, opt.max_bisections)
  @. B = z - zo   # B is free now: use it as work vector
  change = maximum(abs, B)
  StepInfo(true, f, NaN, NaN, change, _step_size(m), max(0.0, vol - V), 1, λ, stat)
end

# Next shift of the linearised constraint from the corrections so far: regula falsi kept inside the bracket once the
# root is bracketed, secant (or the chord s + G) before
function _next_shift(sh, G, lo, hi, prev)
  slo, Glo = lo
  shi, Ghi = hi
  sp, Gp = prev
  if isfinite(slo) && isfinite(shi)
    sn = slo + (shi - slo) * Glo / (Glo - Ghi)
    return clamp(sn, slo + 0.05 * (shi - slo), shi - 0.05 * (shi - slo))
  elseif isfinite(sp) && (G - Gp) * (sh - sp) < 0
    return sh - G * (sh - sp) / (G - Gp)
  end
  sh + G
end

function _oc_step_nonlinear!(opt::OC, s::OCState, z, oracle, f, m)
  g = constraint(oracle, 1, z)
  isfinite(g) || return StepInfo(false, f, NaN, NaN, 0.0, _step_size(m), Inf, 1, NaN, NaN)
  a = constraint_gradient!(s.a, oracle, 1, z)
  amax = maximum(a)
  amax > 0 || throw(ArgumentError("OC: the derivative of the constraint has no positive entry"))
  # entries with a zero derivative (a saturated projection, or a block the constraint does not depend on) cost no
  # volume here: the floor lets them move up to the move limit in the direction of descent
  @. a = max(a, 1e-12 * amax)
  lb, ub = bounds(oracle)
  stat = _oc_stationarity!(s, oracle, z, a, dot(a, z) - g)
  copyto!(s.z_old, z)
  zo, B, η = s.z_old, s.dF, opt.damping
  @. B = _oc_power(max(zero(B), -B) / a, η)
  tol = min(opt.bisection_tol, 1e-7)
  Vlin = dot(a, zo) - g
  sh = s.shift
  _, λ, reached, active = _oc_bisect!(z, zo, B, a, lb, ub, Vlin - sh, m, η, tol, opt.max_bisections)
  G = constraint(oracle, 1, z)
  lo, hi, prev = (-Inf, NaN), (Inf, NaN), (NaN, NaN)
  for _ in 1:opt.max_corrections
    (isfinite(G) && abs(G) > opt.constraint_tol) || break
    (!reached && G > 0) && break   # the move limit does not allow getting closer in this step
    (!active && G < 0) && break    # inactive constraint: the step is the unconstrained one
    G > 0 ? (lo = (sh, G)) : (hi = (sh, G))
    sn = _next_shift(sh, G, lo, hi, prev)
    prev = (sh, G)
    sh = sn
    _, λ, reached, active = _oc_bisect!(z, zo, B, a, lb, ub, Vlin - sh, m, η, tol, opt.max_bisections)
    G = constraint(oracle, 1, z)
  end
  s.shift = isfinite(sh) ? sh : 0.0
  @. B = z - zo
  StepInfo(true, f, NaN, NaN, maximum(abs, B), _step_size(m), isfinite(G) ? max(0.0, G) : Inf, 1, λ, stat)
end

# stationarity measure at z, before the step (s.dF holds ∂F/∂z)
function _oc_stationarity!(s::OCState, oracle, z, a, V)
  riesz!(s.sg, oracle, s.dF)
  riesz!(s.sga, oracle, a)
  stationarity!(s.sw, oracle, z, s.sg, s.sga, a, V)
end

computes_kkt(::OC) = false
