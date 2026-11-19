"""
    ConstraintInfo(lo, hi; linear = false, design_only = false, field = false)

What an optimiser knows about one constraint `lo ≤ g(z) ≤ hi` (values already scaled): `lo = -Inf` for `g ≤ hi`
(the usual `g ≤ 0`), `lo = hi` for an equality, both finite for an interval. `linear`: `g` is affine in z (the
oracle then gives `(a, V)`, [`linear_constraint`](@ref)); `design_only`: evaluating it solves no state; `field`: a
constraint per point (reserved; always `false`).
"""
struct ConstraintInfo
  lo::Float64
  hi::Float64
  linear::Bool
  design_only::Bool
  field::Bool
  function ConstraintInfo(lo::Real, hi::Real; linear::Bool = false, design_only::Bool = false, field::Bool = false)
    (isnan(lo) || isnan(hi) || lo > hi || lo == Inf || hi == -Inf) &&
      throw(ArgumentError("ConstraintInfo: need lo ≤ hi, got lo = $lo, hi = $hi"))
    new(Float64(lo), Float64(hi), linear, design_only, field)
  end
end

"""
    violation(c::ConstraintInfo, g) -> Float64

How far `g` is from `[lo, hi]` (`0` inside).
"""
violation(c::ConstraintInfo, g) = max(g - c.hi, c.lo - g, 0.0)
is_upper(c::ConstraintInfo) = c.lo == -Inf
is_equality(c::ConstraintInfo) = c.lo == c.hi

"""
    AbstractOracle

What an optimiser may ask about the problem. Methods to implement for a subtype `O`:

- `objective(o::O, z)`: value of the objective at `z` (forward problem only). Return `NaN` or `Inf` if the value cannot
  be computed (for example, the forward problem did not converge); the optimisers treat it as a failed trial.
- `gradient!(dF, o::O, z)`: writes the derivative `∂F/∂z` (a dual vector, no metric applied) into `dF`. The optimisers
  only call it for the `z` of their last call to `objective` on this oracle, so an implementation can reuse the
  forward state of that call and only pay for the adjoint.
- `bounds(o::O)`: `(lb, ub)`, the bounds of `z` (scalars or vectors of the length of `z`), with `lb < ub`.
- The constraints `loᵢ ≤ gᵢ(z) ≤ hiᵢ`, `i = 1, …, num_constraints(o)` (default none):
  - `num_constraints(o::O)` and `constraint_info(o::O, i)` (a [`ConstraintInfo`](@ref));
  - linear ones: `linear_constraint(o::O, i)` returns `(a, V)` with `gᵢ(z) = aᵀz − V`; the optimisers then use `a`
    and `V` exactly, and `constraint` and `constraint_gradient!` have defaults built from them;
  - the others: `linear_constraint(o, i)` returns `nothing` and the oracle implements `constraint(o::O, i, z)` (the
    scaled value; `NaN` or `Inf` if it cannot be computed) and `constraint_gradient!(dG, o::O, i, z)` (`∂gᵢ/∂z` at the
    `z` of the last call to `constraint(o, i, ·)`).
  `constraints!(g, o, z)` (all at once, at a trial design) and `weighted_gradient!` (the derivative of
  `w₀ f + Σ wᵢ gᵢ`) have defaults built from those.

Methods with a default (Euclidean metric); they are not exported, extend them as `Optimizers.inner`:

- `inner(o::O, x, y)`: inner product of two primal vectors; default `dot(x, y)`.
- `riesz!(g, o::O, dF)`: primal representative `g = M⁻¹ dF` of a dual vector; default `copyto!(g, dF)`.

Optimisers only use these functions, broadcasting and global reductions (`dot`, `maximum`) on the vectors. If the
bounds, the constraints or the objective change (continuation), call [`reset!`](@ref) on the optimiser.
"""
abstract type AbstractOracle end

"""
    objective(oracle, z)

Value of the objective at `z`. See [`AbstractOracle`](@ref).
"""
function objective end

"""
    gradient!(dF, oracle, z)

Writes `∂F/∂z` into `dF`, for the `z` of the last call to [`objective`](@ref). See [`AbstractOracle`](@ref).
"""
function gradient! end

"""
    bounds(oracle) -> (lb, ub)

Bounds of the design vector. See [`AbstractOracle`](@ref).
"""
function bounds end

"""
    num_constraints(oracle) -> Int

Number of constraints. Default `0`. See [`AbstractOracle`](@ref).
"""
num_constraints(::AbstractOracle) = 0

"""
    constraint_info(oracle, i) -> ConstraintInfo

Description of the constraint `i`. See [`AbstractOracle`](@ref).
"""
function constraint_info end

"""
    linear_constraint(oracle, i) -> (a, V) or nothing

The constraint `i` as `aᵀz − V` when it is linear. Default `nothing`. See [`AbstractOracle`](@ref).
"""
linear_constraint(::AbstractOracle, i) = nothing

"""
    constraint(oracle, i, z) -> Float64

Value of the constraint `i` at `z`. Default, for a linear constraint: `aᵀz − V`. See [`AbstractOracle`](@ref).
"""
function constraint(o::AbstractOracle, i::Integer, z)
  c = linear_constraint(o, i)
  c === nothing && throw(ArgumentError("constraint: the constraint $i is not linear; implement constraint for it"))
  dot(c[1], z) - c[2]
end

"""
    constraint_gradient!(dG, oracle, i, z) -> dG

Writes `∂gᵢ/∂z` into `dG`, for the `z` of the last call to `constraint(oracle, i, ·)`. Default, for a linear
constraint: `a`. See [`AbstractOracle`](@ref).
"""
function constraint_gradient!(dG, o::AbstractOracle, i::Integer, z)
  c = linear_constraint(o, i)
  c === nothing && throw(ArgumentError("constraint_gradient!: the constraint $i is not linear; implement it"))
  copyto!(dG, c[1])
end

"""
    constraints!(g, oracle, z) -> g

Values of all the constraints at `z` (forward solves only, no derivatives): what a method that checks a trial design
needs (GCMMA). Default: `constraint(oracle, i, z)` for each `i`.
"""
function constraints!(g, o::AbstractOracle, z)
  for i in 1:num_constraints(o)
    g[i] = constraint(o, i, z)
  end
  g
end

"""
    weighted_gradient!(dL, oracle, w0, w, z) -> dL

Derivative of `w0 ⋅ f + Σᵢ wᵢ gᵢ` at the `z` of the last calls to `objective` and `constraint`. Default: the separate
derivatives, added (an implementation can do it in one reverse sweep). Zero weights cost nothing.
"""
function weighted_gradient!(dL, o::AbstractOracle, w0, w, z)
  gradient!(dL, o, z)
  dL .*= w0
  any(!iszero, w) || return dL
  work = similar(dL)
  for i in eachindex(w)
    iszero(w[i]) && continue
    constraint_gradient!(work, o, i, z)
    dL .+= w[i] .* work
  end
  dL
end

"""
    inner(oracle, x, y)

Inner product of two primal vectors in the metric of the design. Default: `dot(x, y)`. Not exported (Gridap exports
another `inner`): use `Optimizers.inner`.
"""
inner(::AbstractOracle, x, y) = dot(x, y)

"""
    riesz!(g, oracle, dF)

Writes into `g` the primal representative `M⁻¹ dF` of the dual vector `dF`. Default: `copyto!(g, dF)`.
"""
riesz!(g, ::AbstractOracle, dF) = copyto!(g, dF)

"""
    FunctionOracle(f, df!; lb = 0.0, ub = 1.0, constraint = nothing, constraints = nothing, metric = nothing)

Oracle built from plain functions, for analytical tests and small problems.

- `f(z)`: objective value.
- `df!(dF, z)`: writes `∂F/∂z` into `dF`.
- `constraint = (a, V)` with a vector `a`: linear constraint `aᵀz ≤ V`.
- `constraint = (g, dg!)` with functions: nonlinear constraint `g(z) ≤ 0`, `dg!(dG, z)` writes `∂g/∂z`.
- `constraints = ((g₁, dg₁!, lo₁, hi₁), …)`: several constraints `loᵢ ≤ gᵢ(z) ≤ hiᵢ` (taken as nonlinear), instead of
  `constraint`.
- `metric = m`: diagonal metric, `inner(x, y) = Σ mᵢ xᵢ yᵢ` and `riesz!(g, dF) = dF ./ m` (cell volumes for P0
  designs, lumped mass for P1 designs). `nothing` is the Euclidean metric.

`evaluations(o)` returns `(value, derivative)`, the number of calls of each kind.
"""
mutable struct FunctionOracle{F,G,L,U,C,M} <: AbstractOracle
  f::F
  df!::G
  lb::L
  ub::U
  constraint::C
  metric::M
  nvalue::Int
  nderivative::Int
end

# several constraints (g, dg!, lo, hi)
struct _Constraints{T<:Tuple}
  c::T
  infos::Vector{ConstraintInfo}
end

function FunctionOracle(f, df!; lb = 0.0, ub = 1.0, constraint = nothing, constraints = nothing, metric = nothing)
  constraint === nothing || constraint isa Tuple{AbstractVector,Real} || constraint isa Tuple{Function,Function} ||
    throw(ArgumentError("FunctionOracle: constraint must be (a, V) or (g, dg!)"))
  if constraints !== nothing
    constraint === nothing || throw(ArgumentError("FunctionOracle: give constraint or constraints, not both"))
    (constraints isa Tuple && all(c -> c isa Tuple{Any,Any,Real,Real}, constraints)) ||
      throw(ArgumentError("FunctionOracle: constraints must be a tuple of (g, dg!, lo, hi)"))
    constraint = _Constraints(constraints, [ConstraintInfo(c[3], c[4]; design_only = true) for c in constraints])
  end
  FunctionOracle(f, df!, lb, ub, constraint, metric, 0, 0)
end

function objective(o::FunctionOracle, z)
  o.nvalue += 1
  o.f(z)
end

function gradient!(dF, o::FunctionOracle, z)
  o.nderivative += 1
  o.df!(dF, z)
  dF
end

bounds(o::FunctionOracle) = (o.lb, o.ub)
num_constraints(o::FunctionOracle) = o.constraint === nothing ? 0 : 1
function constraint_info(o::FunctionOracle, i)
  (o.constraint !== nothing && i == 1) || throw(BoundsError(o, i))
  ConstraintInfo(-Inf, 0.0; linear = o.constraint isa Tuple{AbstractVector,Real}, design_only = true)
end
linear_constraint(o::FunctionOracle, i) = o.constraint isa Tuple{AbstractVector,Real} ? o.constraint : nothing
constraint(o::FunctionOracle{F,G,L,U,<:Tuple{Function,Function}}, i::Integer, z) where {F,G,L,U} = o.constraint[1](z)
constraint_gradient!(dG, o::FunctionOracle{F,G,L,U,<:Tuple{Function,Function}}, i::Integer, z) where {F,G,L,U} =
  (o.constraint[2](dG, z); dG)

num_constraints(o::FunctionOracle{F,G,L,U,<:_Constraints}) where {F,G,L,U} = length(o.constraint.c)
constraint_info(o::FunctionOracle{F,G,L,U,<:_Constraints}, i) where {F,G,L,U} = o.constraint.infos[i]
linear_constraint(o::FunctionOracle{F,G,L,U,<:_Constraints}, i) where {F,G,L,U} = nothing
constraint(o::FunctionOracle{F,G,L,U,<:_Constraints}, i::Integer, z) where {F,G,L,U} = o.constraint.c[i][1](z)
constraint_gradient!(dG, o::FunctionOracle{F,G,L,U,<:_Constraints}, i::Integer, z) where {F,G,L,U} =
  (o.constraint.c[i][2](dG, z); dG)

inner(o::FunctionOracle{F,G,L,U,C,Nothing}, x, y) where {F,G,L,U,C} = dot(x, y)
inner(o::FunctionOracle{F,G,L,U,C,<:Vector{Float64}}, x::Vector{Float64}, y::Vector{Float64}) where {F,G,L,U,C} =
  sum(i -> o.metric[i] * x[i] * y[i], eachindex(x, y, o.metric))
# any other vector type: a lazy broadcast reduced without indexing (no allocation)
inner(o::FunctionOracle, x, y) = sum(Broadcast.instantiate(Broadcast.broadcasted(*, o.metric, x, y)))

riesz!(g, o::FunctionOracle{F,G,L,U,C,Nothing}, dF) where {F,G,L,U,C} = copyto!(g, dF)
riesz!(g, o::FunctionOracle, dF) = (g .= dF ./ o.metric)

evaluations(o::FunctionOracle) = (o.nvalue, o.nderivative)

"""
    stationarity!(w, oracle, z, g, ga, a, V) -> S

Stationarity measure of the paper of SiMPL (Kim, Lazarov, Surowiec, Keith, 2024): `S = ‖z − P(z − g)‖_M`, with
`g = M⁻¹ ∂F/∂z` (see [`riesz!`](@ref)), `M` the metric of the oracle and `P` the projection, in that metric, onto the
bounds and the constraint `aᵀz ≤ V` (`a = nothing`: bounds only; a nonlinear constraint is linearised at `z`). For a
diagonal metric `P(y) = clamp(y − c M⁻¹a, lb, ub)` with the smallest `c ≥ 0` that meets the constraint, found by
bisection; `ga = M⁻¹a` (pass `riesz!(ga, oracle, a)`, not `a`). `w` is a work vector. It is absolute: it depends on
the scale of the objective and on the metric (with the Euclidean metric it depends on the mesh). Returns `NaN` when no
point within the bounds meets the constraint (`V` below the minimum of `aᵀz` over the bounds) or when `V` is not
finite. Only broadcasting and reductions: no allocations.
"""
function stationarity!(w, oracle, z, g, ga, a, V)
  lb, ub = bounds(oracle)
  @. w = clamp(z - g, lb, ub)
  if a !== nothing
    isfinite(V) || return NaN
    if dot(a, w) > V
      φ(c) = (@. w = clamp(z - g - c * ga, lb, ub); dot(a, w) - V)
      lo, hi = 0.0, 1.0
      n = 0
      while φ(hi) > 0 && n < 200
        lo, hi = hi, 2hi
        n += 1
      end
      φ(hi) > 0 && return NaN                     # unreachable within the bounds
      for _ in 1:200
        mid = (lo + hi) / 2
        φ(mid) > 0 ? (lo = mid) : (hi = mid)
        hi - lo <= 1e-12 * max(hi, 1e-300) && break
      end
      φ(hi)
    end
  end
  @. w = w - z
  sqrt(max(inner(oracle, w, w), 0.0))
end
