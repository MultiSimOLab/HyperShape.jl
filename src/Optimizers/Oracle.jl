"""
    AbstractOracle

What an optimiser may ask about the problem. Methods to implement for a subtype `O`:

- `objective(o::O, z)`: value of the objective at `z` (forward problem only). Return `NaN` or `Inf` if the value cannot
  be computed (for example, the forward problem did not converge); the optimisers treat it as a failed trial.
- `gradient!(dF, o::O, z)`: writes the derivative `∂F/∂z` (a dual vector, no metric applied) into `dF`. The optimisers
  only call it for the `z` of their last call to `objective` on this oracle, so an implementation can reuse the
  forward state of that call and only pay for the adjoint.
- `bounds(o::O)`: `(lb, ub)`, the bounds of `z` (scalars or vectors of the length of `z`), with `lb < ub`.
- `linear_constraint(o::O)`: `(a, V)` for the constraint `aᵀz ≤ V` (`a ≥ 0` is a dual vector, for example the
  derivative of the volume), or `nothing` if there is none.

Methods with a default (Euclidean metric); they are not exported, extend them as `Optimizers.inner`:

- `inner(o::O, x, y)`: inner product of two primal vectors; default `dot(x, y)`.
- `riesz!(g, o::O, dF)`: primal representative `g = M⁻¹ dF` of a dual vector; default `copyto!(g, dF)`.

Optimisers only use these functions, broadcasting and global reductions (`dot`, `maximum`) on the vectors. If the
bounds, the constraint or the objective change (continuation), call [`reset!`](@ref) on the optimiser.
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
    linear_constraint(oracle) -> (a, V) or nothing

Linear constraint `aᵀz ≤ V`. See [`AbstractOracle`](@ref).
"""
function linear_constraint end

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
    FunctionOracle(f, df!; lb = 0.0, ub = 1.0, constraint = nothing, metric = nothing)

Oracle built from plain functions, for analytical tests and small problems.

- `f(z)`: objective value.
- `df!(dF, z)`: writes `∂F/∂z` into `dF`.
- `constraint = (a, V)`: linear constraint `aᵀz ≤ V`.
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

function FunctionOracle(f, df!; lb = 0.0, ub = 1.0, constraint = nothing, metric = nothing)
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
linear_constraint(o::FunctionOracle) = o.constraint

inner(o::FunctionOracle{F,G,L,U,C,Nothing}, x, y) where {F,G,L,U,C} = dot(x, y)
inner(o::FunctionOracle, x, y) = sum(i -> o.metric[i] * x[i] * y[i], eachindex(x, y, o.metric))

riesz!(g, o::FunctionOracle{F,G,L,U,C,Nothing}, dF) where {F,G,L,U,C} = copyto!(g, dF)
riesz!(g, o::FunctionOracle, dF) = (g .= dF ./ o.metric)

evaluations(o::FunctionOracle) = (o.nvalue, o.nderivative)
