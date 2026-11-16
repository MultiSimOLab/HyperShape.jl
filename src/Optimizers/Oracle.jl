"""
    AbstractOracle

What an optimiser may ask about the problem. Methods to implement for a subtype `O`:

- `objective(o::O, z)`: value of the objective at `z` (forward problem only). Return `NaN` or `Inf` if the value cannot
  be computed (for example, the forward problem did not converge); the optimisers treat it as a failed trial.
- `gradient!(dF, o::O, z)`: writes the derivative `∂F/∂z` (a dual vector, no metric applied) into `dF`. The optimisers
  only call it for the `z` of their last call to `objective` on this oracle, so an implementation can reuse the
  forward state of that call and only pay for the adjoint.
- `bounds(o::O)`: `(lb, ub)`, the bounds of `z` (scalars or vectors of the length of `z`), with `lb < ub`.
- The constraint `g(z) ≤ 0`, if there is one (at most one), in either of two ways:
  - linear: `linear_constraint(o::O)` returns `(a, V)` for `aᵀz ≤ V` (`a ≥ 0` is a dual vector, for example the
    derivative of the volume); the optimisers then use `a` and `V` exactly;
  - nonlinear: `linear_constraint(o::O)` returns `nothing` and the oracle implements `has_constraint(o::O) = true`,
    `constraint(o::O, z)` (the value of `g`, scaled, `≤ 0` when feasible; `NaN` or `Inf` if it cannot be computed)
    and `constraint_gradient!(dG, o::O, z)` (`∂g/∂z` at the `z` of the last call to `constraint`).
  For a linear constraint the last three have defaults built from `(a, V)`.

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
    has_constraint(oracle) -> Bool

Whether the problem has a constraint. Default: `linear_constraint(oracle) !== nothing`. See [`AbstractOracle`](@ref).
"""
has_constraint(o::AbstractOracle) = linear_constraint(o) !== nothing

"""
    constraint(oracle, z) -> Float64

Value of the constraint `g(z)` (`≤ 0` when feasible). Default, for a linear constraint: `aᵀz - V`. See
[`AbstractOracle`](@ref).
"""
function constraint(o::AbstractOracle, z)
  c = linear_constraint(o)
  c === nothing && throw(ArgumentError("constraint: the oracle has no linear constraint; implement constraint for it"))
  dot(c[1], z) - c[2]
end

"""
    constraint_gradient!(dG, oracle, z) -> dG

Writes `∂g/∂z` into `dG`, for the `z` of the last call to [`constraint`](@ref). Default, for a linear constraint:
`a`. See [`AbstractOracle`](@ref).
"""
function constraint_gradient!(dG, o::AbstractOracle, z)
  c = linear_constraint(o)
  c === nothing && throw(ArgumentError("constraint_gradient!: the oracle has no linear constraint; implement it"))
  copyto!(dG, c[1])
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
    FunctionOracle(f, df!; lb = 0.0, ub = 1.0, constraint = nothing, metric = nothing)

Oracle built from plain functions, for analytical tests and small problems.

- `f(z)`: objective value.
- `df!(dF, z)`: writes `∂F/∂z` into `dF`.
- `constraint = (a, V)` with a vector `a`: linear constraint `aᵀz ≤ V`.
- `constraint = (g, dg!)` with functions: nonlinear constraint `g(z) ≤ 0`, `dg!(dG, z)` writes `∂g/∂z`.
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
  constraint === nothing || constraint isa Tuple{AbstractVector,Real} || constraint isa Tuple{Function,Function} ||
    throw(ArgumentError("FunctionOracle: constraint must be (a, V) or (g, dg!)"))
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
linear_constraint(o::FunctionOracle) = o.constraint isa Tuple{AbstractVector,Real} ? o.constraint : nothing
has_constraint(o::FunctionOracle) = o.constraint !== nothing
constraint(o::FunctionOracle{F,G,L,U,<:Tuple{Function,Function}}, z) where {F,G,L,U} = o.constraint[1](z)
constraint_gradient!(dG, o::FunctionOracle{F,G,L,U,<:Tuple{Function,Function}}, z) where {F,G,L,U} =
  (o.constraint[2](dG, z); dG)

inner(o::FunctionOracle{F,G,L,U,C,Nothing}, x, y) where {F,G,L,U,C} = dot(x, y)
inner(o::FunctionOracle, x, y) = sum(i -> o.metric[i] * x[i] * y[i], eachindex(x, y, o.metric))

riesz!(g, o::FunctionOracle{F,G,L,U,C,Nothing}, dF) where {F,G,L,U,C} = copyto!(g, dF)
riesz!(g, o::FunctionOracle, dF) = (g .= dF ./ o.metric)

evaluations(o::FunctionOracle) = (o.nvalue, o.nderivative)
