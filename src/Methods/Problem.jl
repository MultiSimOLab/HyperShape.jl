# Problem, metric and the oracle that connects the functionals with the optimisers.

_optvariables(F::Functional) = get_optvariables(get_designfields(F.θ))

struct _Validated end   # only the keyword constructors build problems

"""
    Optimization(; objective, constraints = (;))
    Optimization(z; objective, constraints = (;))

Optimisation problem: minimise the [`Functional`](@ref) `objective` over the variables `z` behind it (given or not;
when given they are checked) subject to the bounds of `z` and the named `constraints` `g(z) ≤ 0`, for example
`constraints = (volume = Vol,)`. The names go to the history (`history.constraints.volume`), the banner and the
checkpoint. In phase 0, at most one
constraint, a design-only functional: typically [`Volume`](@ref HyperShape.Functionals.Volume)`(θ, dΩ; max_fraction)`
(volume of the physical density, nonlinear in z with a tanh projection) or `Volume(x, dΩ; max_fraction)` (volume of
the design density, linear).

A constraint built as linear (`Volume` over identity or filter maps, or a `Functional(…; linear = true)`) is checked
to be linear when the optimisation starts, and costs nothing in the loop (its vector is computed once). A nonlinear
one is linearised at every step by the optimiser, which then corrects it with evaluations of the true constraint
(one filter solve each, no forward solve): see `OC` and `SiMPL`.

The problem owns the context `ctx` ([`OptContext`](@ref)) that reaches the solve functions of the user.
"""
struct Optimization{Z<:OptVariables,J<:Functional,C<:NamedTuple}
  z::Z
  objective::J
  constraints::C
  ctx::OptContext
  Optimization(::_Validated, z, J, cs) = new{typeof(z),typeof(J),typeof(cs)}(z, J, cs, OptContext(:optimization))
end

Optimization(z::OptVariables; objective, constraints = (;)) = _optimization(z, objective, constraints)
Optimization(; objective, constraints = (;)) = _optimization(nothing, objective, constraints)

function _optimization(z, objective, constraints)
  objective isa Functional || throw(ArgumentError("Optimization: the objective must be a Functional"))
  zJ = _optvariables(objective)
  z === nothing && (z = zJ)
  zJ === z || throw(ArgumentError("Optimization: the objective is not built on these variables z"))
  constraints isa Functional &&
    throw(ArgumentError("Optimization: name the constraints, for example constraints = (volume = Vol,)"))
  constraints isa NamedTuple ||
    throw(ArgumentError("Optimization: constraints must be a named tuple, for example (volume = Vol,)"))
  length(constraints) <= 1 || throw(ArgumentError("Optimization: at most one constraint in phase 0"))
  for (n, c) in pairs(constraints)
    n === :objective && throw(ArgumentError("Optimization: objective is not a valid name for a constraint"))
    c isa Functional || throw(ArgumentError("Optimization: the constraint $n must be a Functional"))
    c.S === nothing || throw(ArgumentError("Optimization: the constraint $n must be a design-only Functional (no states) in phase 0"))
    _optvariables(c) === z || throw(ArgumentError("Optimization: the constraint $n is not built on these variables z"))
  end
  Optimization(_Validated(), z, objective, constraints)
end

Base.show(io::IO, p::Optimization) =
  print(io, "Optimization(", length(p.z), " variables; constraints: ",
        isempty(p.constraints) ? "none" : join(keys(p.constraints), ", "), ")")

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
    DensityMethod(optimizer; metric = default_metric(optimizer))

Density method (SIMP, RAMP or any interpolation written by the user in the weak forms): the step is `optimizer` (any
`Optimizer`: `OC(…)`, `SiMPL(…)`). `metric`: [`LumpedMass`](@ref)`(dΩ)`, [`Euclidean`](@ref)`()`, a vector of
positive weights (one per entry of z: `inner(x, y) = Σ mᵢ xᵢ yᵢ`), an object of the user with a method
[`metric_weights`](@ref)`(m, z) -> Vector` (called once, at the start of `solve!`), or a named tuple with one of them
per block of z. SiMPL uses
it in its step and its optimality measure, and needs it (no default). OC uses it only for the stationarity measure
(see [`StopCriteria`](@ref)); its default is `Euclidean()`, which makes that measure depend on the mesh.
"""
struct DensityMethod{O<:Optimizer,M}
  optimizer::O
  metric::M
  default_metric::Bool   # the metric was not given (shown in the banner)
  function DensityMethod(optimizer::Optimizer; metric = nothing)
    given = metric !== nothing
    given || (metric = default_metric(optimizer))
    ok = _valid_metric(metric) || (metric isa NamedTuple && all(_valid_metric, values(metric)))
    ok || throw(ArgumentError("DensityMethod: metric must be LumpedMass(dΩ), Euclidean(), a vector of positive weights, " *
                              "an object with a metric_weights(m, z) method, or a named tuple of them per block"))
    new{typeof(optimizer),typeof(metric)}(optimizer, metric, !given)
  end
end

"""
    default_metric(optimizer)

Metric of [`DensityMethod`](@ref) when none is given: `Euclidean()` for OC (which only uses it for the stationarity measure); an error for any other
optimiser, which must be given one.
"""
default_metric(::OC) = Euclidean()
default_metric(opt::Optimizer) =
  throw(ArgumentError("DensityMethod: $(nameof(typeof(opt))) needs a metric: DensityMethod(opt; metric = LumpedMass(dΩ)) or Euclidean()"))

# an object of the user is checked when the weights are asked for (its metric_weights may restrict the type of z)
_valid_metric(m) = !(m isa Union{Nothing,Symbol,AbstractString,Number,Tuple,AbstractVector}) ||
                   m isa AbstractVector{<:Real}

"""
    metric_weights(m, z) -> AbstractVector

Weights of a diagonal metric of the user, one per entry of the variables `z` (or of the block, in a named tuple of
metrics: then `z` is the space of that block). Called once, at the start of [`solve!`](@ref); they must be positive.
"""
function metric_weights end

_block_metric(::Euclidean, V) = ones(num_free_dofs(V))
_block_metric(m::AbstractVector{<:Real}, V) = _checked_weights(m, num_free_dofs(V))
_block_metric(m, V) = _checked_weights(_weights(m, V), num_free_dofs(V))

function _weights(m, z)
  applicable(metric_weights, m, z) ||
    throw(ArgumentError("DensityMethod: no method metric_weights(::$(typeof(m)), ::$(typeof(z))) for the metric"))
  metric_weights(m, z)
end

function _checked_weights(w, n)
  length(w) == n || throw(ArgumentError("DensityMethod: the metric has $(length(w)) weights; $n are needed"))
  all(x -> isfinite(x) && x > 0, w) || throw(ArgumentError("DensityMethod: the weights of the metric must be positive and finite"))
  collect(Float64, w)
end
_block_metric(m::LumpedMass, V) = collect(assemble_vector(v -> ∫(v)m.measure, V))

_metric_vector(::Euclidean, z::OptVariables) = nothing
_metric_vector(m::LumpedMass, z::OptVariables) = reduce(vcat, [_block_metric(m, b.space) for b in z.blocks])
_metric_vector(m::AbstractVector{<:Real}, z::OptVariables) = _checked_weights(m, length(z))
_metric_vector(m, z::OptVariables) = _checked_weights(_weights(m, z), length(z))
function _metric_vector(m::NamedTuple, z::OptVariables)
  Set(keys(m)) == Set(z.names) ||
    throw(ArgumentError("DensityMethod: the metric has blocks $(join(keys(m), ", ")); z has $(join(z.names, ", "))"))
  all(x -> x isa Euclidean, values(m)) && return nothing
  reduce(vcat, [_block_metric(getproperty(m, n), b.space) for (n, b) in zip(z.names, z.blocks)])
end

"""
    FunctionalOracle

Oracle (see `HyperShape.Optimizers.AbstractOracle`) of an [`Optimization`](@ref): `objective` and `gradient!` call the
objective functional at the requested design; a linear constraint gives `(a, V)` computed once at the start
(`linear_constraint`), a nonlinear one is evaluated at the requested design (`constraint`, `constraint_gradient!`:
the fields only, no forward solve); `inner`/`riesz!` use the metric of the method.

- It is the only writer of the variables `z`; a design equal to the current one is not set again, so the value cache
  of the functional avoids repeated forward solves.
- It keeps a copy of the design, value and derivative of its last `gradient!`: after a rejected step, the optimiser
  asks again at that design and gets them without solves.
- A gradient that is not finite throws [`OptimizationFailure`](@ref).
"""
mutable struct FunctionalOracle{J,C,Z,M} <: AbstractOracle
  J::J
  C::C                          # the constraint (a Functional) or nothing
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
  m === nothing || all(>(0), m) || throw(ArgumentError("DensityMethod: the metric has non-positive entries"))
  C = isempty(problem.constraints) ? nothing : only(problem.constraints)
  FunctionalOracle(problem.objective, C, z, problem.ctx, copy(get_values(z)), copy(lb), copy(ub), nothing, 0.0, m,
                   zeros(n), zeros(n), NaN, false)
end

_invalidate_snapshot!(o::FunctionalOracle) = (o.snap_valid = false; o)

function _set_design!(o::FunctionalOracle, v)
  get_values(o.z) == v || set_values!(o.z, v)
  nothing
end

# Optimisers may hand back the current design through their own variables (SiMPL: a sigmoid of a logit), equal up to
# rounding: such a design is taken as the current one, so it is neither solved again nor treated as a trial.
function _near(v, w)
  @inbounds for i in eachindex(v, w)
    abs(v[i] - w[i]) <= 1e-10 * max(abs(w[i]), 1.0) || return false   # SiMPL keeps 1e-12 away from the bounds
  end
  true
end
_canonical(o::FunctionalOracle, v) = (v === o.zk || _near(v, o.zk)) ? o.zk : v

function objective(o::FunctionalOracle, v)
  ctx = o.ctx
  v = _canonical(o, v)
  set_context!(ctx, :trial, ctx.trial + 1)
  set_context!(ctx, :phase, v === o.zk ? :optimization : :linesearch)
  (o.snap_valid && v == o.snap_z) && return o.snap_f
  _set_design!(o, v)
  functional_value(o.J, ctx)
end

# the variables, the fields and the forward state at the loop's design (not served from the snapshot)
function _sync!(o::FunctionalOracle)
  set_context!(o.ctx, :phase, :optimization)
  _set_design!(o, o.zk)
  functional_value(o.J, o.ctx)
end

function gradient!(dF, o::FunctionalOracle, v)
  v = _canonical(o, v)
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
has_constraint(o::FunctionalOracle) = o.C !== nothing
_is_linear(o::FunctionalOracle) = o.C !== nothing && o.C.linear

function constraint(o::FunctionalOracle, v)
  o.a === nothing || return dot(o.a, v) - o.V
  v = _canonical(o, v)
  _set_design!(o, v)
  functional_value(o.C, o.ctx)
end

function constraint_gradient!(dG, o::FunctionalOracle, v)
  o.a === nothing || return copyto!(dG, o.a)
  v = _canonical(o, v)
  get_values(o.z) == v ||
    throw(ArgumentError("FunctionalOracle: constraint_gradient! asked at a design different from the last constraint"))
  functional_gradient!(dG, o.C, o.ctx)
  all(isfinite, dG) || throw(OptimizationFailure("the gradient of the constraint is not finite"))
  dG
end

inner(o::FunctionalOracle{J,C,Z,Nothing}, x, y) where {J,C,Z} = dot(x, y)
function inner(o::FunctionalOracle, x, y)
  m = o.metric
  s = zero(eltype(m))
  @inbounds @simd for i in eachindex(m, x, y)
    s += m[i] * x[i] * y[i]
  end
  s
end

riesz!(g, o::FunctionalOracle{J,C,Z,Nothing}, dF) where {J,C,Z} = copyto!(g, dF)
riesz!(g, o::FunctionalOracle, dF) = (g .= dF ./ o.metric)

# a and V of the linear constraint g(z) = aᵀz − V ≤ 0 (scaled as the functional), with a check of linearity at four
# designs that do not depend on the starting one (relative tolerance 1e-6, so a filter solved iteratively passes)
function _setup_constraint!(o::FunctionalOracle, c::Functional, ctx)
  z = o.z
  z0 = copy(get_values(z))
  set_context!(ctx, :phase, :setup)
  a = functional_gradient!(zeros(length(z0)), c, ctx)
  v0 = functional_value(c, ctx)
  (isfinite(v0) && all(isfinite, a)) || throw(ArgumentError("Optimization: the linear constraint is not finite at the start"))
  V = dot(a, z0) - v0
  w = o.ub .- o.lb
  # oscillating designs and uniform shifts (a filter damps the oscillations but not the shifts)
  probes = (o.lb .+ w .* [0.5 + 0.25 * sin(1.7i) for i in eachindex(z0)],
            o.lb .+ w .* [0.5 + 0.25 * cos(2.3i) for i in eachindex(z0)],
            o.lb .+ 0.25 .* w, o.lb .+ 0.8 .* w)
  for z1 in probes
    set_values!(z, z1)
    v1 = functional_value(c, ctx)
    pred = dot(a, z1) - V
    tol = 1e-6 * (abs(v0) + abs(v1) + sum(i -> abs(a[i] * (z1[i] - z0[i])), eachindex(a)))
    if !(abs(v1 - pred) <= tol)
      set_values!(z, z0)
      throw(ArgumentError("Optimization: the constraint was declared linear but is not linear in z (value $(v1) " *
        "where the linearisation gives $(pred)); declare it with linear = false"))
    end
  end
  set_values!(z, z0)
  o.a, o.V = a, V
  o
end

_constraint_values(o::FunctionalOracle, zk) = o.C === nothing ? () : (constraint(o, zk),)
_multipliers(o::FunctionalOracle, info) = o.C === nothing ? () : (info.multiplier,)

Base.show(io::IO, m::DensityMethod) = print(io, "DensityMethod(", nameof(typeof(m.optimizer)), ")")
