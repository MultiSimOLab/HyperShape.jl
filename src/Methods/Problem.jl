# Problem, metric and the oracle that connects the functionals with the optimisers.

"""
    OptimizationFailure(msg)

Thrown by [`solve!`](@ref) when the optimisation cannot continue: the forward problem does not converge at the
current design (first iteration or right after a continuation jump), every attempt of a step was rejected, or a
gradient is not finite (failed adjoint). `z`, the fields and the states are left at the last accepted design, and
the checkpoint (if any) is written first.
"""
struct OptimizationFailure <: Exception
  msg::String
end
Base.showerror(io::IO, e::OptimizationFailure) = print(io, "OptimizationFailure: ", e.msg)

_optvariables(F::Functional) = get_optvariables(get_designfields(F.θ))

"""
    Optimization(z; objective, linear_constraints = ())

Optimisation problem over the variables `z`: minimise the [`Functional`](@ref) `objective` subject to the bounds of
`z` and, in phase 0, at most one linear constraint `g(z) ≤ 0`, given as a design-only functional (for example the
volume of an unprojected density, `Functional(θ -> sum(∫(θ.ρ)dΩ) - Vmax, θ -> (δ -> ∫(δ)dΩ), θv; scale = Vmax)` with
`θv = PhysicalFields(x; ρ = Identity())`). Its
linearity in z is checked when the optimisation starts. Nonlinear constraints (`constraints = …`) come in phase 1.

The problem owns the context `ctx` ([`OptContext`](@ref)) that reaches the solve functions of the user.
"""
struct Optimization{Z<:OptVariables,J<:Functional,C<:Tuple}
  z::Z
  objective::J
  linear_constraints::C
  ctx::OptContext
end

function Optimization(z::OptVariables; objective, linear_constraints = (), constraints = nothing)
  constraints === nothing ||
    throw(ArgumentError("Optimization: nonlinear constraints come in phase 1; give a linear constraint with linear_constraints"))
  objective isa Functional || throw(ArgumentError("Optimization: the objective must be a Functional"))
  cs = linear_constraints isa Functional ? (linear_constraints,) : Tuple(linear_constraints)
  length(cs) <= 1 || throw(ArgumentError("Optimization: at most one linear constraint in phase 0"))
  _optvariables(objective) === z || throw(ArgumentError("Optimization: the objective is not built on these variables z"))
  for c in cs
    c isa Functional || throw(ArgumentError("Optimization: a linear constraint must be a Functional"))
    c.S === nothing || throw(ArgumentError("Optimization: a linear constraint must be a design-only Functional (no states)"))
    _optvariables(c) === z || throw(ArgumentError("Optimization: a constraint is not built on these variables z"))
  end
  Optimization(z, objective, cs, OptContext(:optimization))
end

"""
    Euclidean()

Euclidean metric of the design vector: `inner(x, y) = xᵀy`. Mesh-dependent on non-uniform meshes.
"""
struct Euclidean end

"""
    LumpedMass(dΩ)

Diagonal metric `m = ∫ v dΩ` assembled on each design space: the cell volumes for a P0 design, the lumped mass (row
sums of the mass matrix) for a P1 design. `inner(x, y) = Σ mᵢ xᵢ yᵢ`, `riesz!(g, dF) = dF ./ m`.
"""
struct LumpedMass{M}
  measure::M
end

const Metric = Union{Euclidean,LumpedMass}

"""
    SIMP(optimizer; metric)

Density method: the step is `optimizer` (any `Optimizer`: `OC(…)`, `SiMPL(…)`). `metric` is mandatory:
[`LumpedMass`](@ref)`(dΩ)` or [`Euclidean`](@ref)`()`, or a named tuple with one of them per block of z. OC
ignores it; SiMPL uses it in its step and its optimality measure.
"""
struct SIMP{O<:Optimizer,M}
  optimizer::O
  metric::M
end

function SIMP(optimizer::Optimizer; metric)
  ok = metric isa Metric || (metric isa NamedTuple && all(m -> m isa Metric, values(metric)))
  ok || throw(ArgumentError("SIMP: metric must be LumpedMass(dΩ), Euclidean() or a named tuple of them per block"))
  SIMP(optimizer, metric)
end

_block_metric(::Euclidean, V) = ones(num_free_dofs(V))
_block_metric(m::LumpedMass, V) = collect(assemble_vector(v -> ∫(v)m.measure, V))

_metric_vector(::Euclidean, z::OptVariables) = nothing
_metric_vector(m::LumpedMass, z::OptVariables) = reduce(vcat, [_block_metric(m, b.space) for b in z.blocks])
function _metric_vector(m::NamedTuple, z::OptVariables)
  Set(keys(m)) == Set(z.names) ||
    throw(ArgumentError("SIMP: the metric has blocks $(join(keys(m), ", ")); z has $(join(z.names, ", "))"))
  all(x -> x isa Euclidean, values(m)) && return nothing
  reduce(vcat, [_block_metric(getproperty(m, n), b.space) for (n, b) in zip(z.names, z.blocks)])
end

"""
    FunctionalOracle

Oracle (see `HyperShape.Optimizers.AbstractOracle`) of an [`Optimization`](@ref): `objective` and `gradient!` call the
objective functional at the requested design, `linear_constraint` gives `(a, V)` computed once at the start, and
`inner`/`riesz!` use the metric of the method.

- It is the only writer of the variables `z`; a design equal to the current one is not set again, so the value cache
  of the functional avoids repeated forward solves.
- It keeps a copy of the design, value and derivative of its last `gradient!`: after a rejected step, the optimiser
  asks again at that design and gets them without solves.
- A gradient that is not finite throws [`OptimizationFailure`](@ref).
"""
mutable struct FunctionalOracle{J,Z,M} <: AbstractOracle
  J::J
  z::Z
  ctx::OptContext
  zk::Vector{Float64}           # the loop's current design (shared with the loop)
  lb::Vector{Float64}
  ub::Vector{Float64}
  a::Union{Nothing,Vector{Float64}}
  V::Float64
  metric::M                     # nothing (Euclidean) or the diagonal metric
  snap_z::Vector{Float64}
  snap_dF::Vector{Float64}
  snap_f::Float64
  snap_valid::Bool
end

function FunctionalOracle(problem::Optimization, metric)
  z = problem.z
  n = length(z)
  lb, ub = get_bounds(z)
  m = _metric_vector(metric, z)
  m === nothing || all(>(0), m) || throw(ArgumentError("SIMP: the metric has non-positive entries"))
  FunctionalOracle(problem.objective, z, problem.ctx, copy(get_values(z)), copy(lb), copy(ub), nothing, 0.0, m,
                   zeros(n), zeros(n), NaN, false)
end

_invalidate_snapshot!(o::FunctionalOracle) = (o.snap_valid = false; o)

function _set_design!(o::FunctionalOracle, v)
  get_values(o.z) == v || set_values!(o.z, v)
  nothing
end

function objective(o::FunctionalOracle, v)
  ctx = o.ctx
  set_context!(ctx, :trial, ctx.trial + 1)
  set_context!(ctx, :phase, v == o.zk ? :optimization : :linesearch)
  (o.snap_valid && v == o.snap_z) && return o.snap_f
  _set_design!(o, v)
  functional_value(o.J, ctx)
end

function gradient!(dF, o::FunctionalOracle, v)
  if o.snap_valid && v == o.snap_z
    copyto!(dF, o.snap_dF)
    return dF
  end
  get_values(o.z) == v ||
    throw(ArgumentError("FunctionalOracle: gradient! asked at a design different from the last objective"))
  set_context!(o.ctx, :phase, :optimization)
  functional_gradient!(dF, o.J, o.ctx)
  all(isfinite, dF) ||
    throw(OptimizationFailure("the gradient of the objective is not finite (did the adjoint state converge?)"))
  copyto!(o.snap_z, v)
  copyto!(o.snap_dF, dF)
  o.snap_f = functional_value(o.J, o.ctx)       # cached: same design
  o.snap_valid = true
  dF
end

bounds(o::FunctionalOracle) = (o.lb, o.ub)
linear_constraint(o::FunctionalOracle) = o.a === nothing ? nothing : (o.a, o.V)

inner(o::FunctionalOracle{J,Z,Nothing}, x, y) where {J,Z} = dot(x, y)
function inner(o::FunctionalOracle, x, y)
  m = o.metric
  s = zero(eltype(m))
  @inbounds @simd for i in eachindex(m, x, y)
    s += m[i] * x[i] * y[i]
  end
  s
end

riesz!(g, o::FunctionalOracle{J,Z,Nothing}, dF) where {J,Z} = copyto!(g, dF)
riesz!(g, o::FunctionalOracle, dF) = (g .= dF ./ o.metric)

# a and V of the linear constraint g(z) = aᵀz − V ≤ 0 (scaled as the functional), with a check of linearity
function _setup_constraint!(o::FunctionalOracle, c::Functional, ctx)
  z = o.z
  z0 = copy(get_values(z))
  set_context!(ctx, :phase, :optimization)
  a = functional_gradient!(zeros(length(z0)), c, ctx)
  v0 = functional_value(c, ctx)
  (isfinite(v0) && all(isfinite, a)) || throw(ArgumentError("Optimization: the linear constraint is not finite at the start"))
  V = dot(a, z0) - v0
  # a second point inside the bounds
  z1 = @. 0.5 * z0 + 0.5 * (o.lb + 0.3 * (o.ub - o.lb))
  if z1 != z0
    set_values!(z, z1)
    v1 = functional_value(c, ctx)
    set_values!(z, z0)
    pred = dot(a, z1) - V
    tol = 1e-8 * (abs(v0) + abs(v1) + sum(i -> abs(a[i] * (z1[i] - z0[i])), eachindex(a)))
    abs(v1 - pred) <= tol || throw(ArgumentError("Optimization: the linear constraint is not linear in z " *
      "(value $(v1) where the linearisation gives $(pred)); a projected volume is a nonlinear constraint (phase 1)"))
  end
  o.a, o.V = a, V
  o
end

_constraint_value(o::FunctionalOracle, zk) = o.a === nothing ? NaN : dot(o.a, zk) - o.V
