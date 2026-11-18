# Problem, metric and the oracle that connects the functionals with the optimisers.

_optvariables(F::AbstractFunctional) = get_optvariables(get_designfields(physical_fields(F)))
_optvariables(c::Constraint) = _optvariables(c.F)

struct _Validated end   # only the keyword constructors build problems

"""
    Optimization(; objective, constraints = (;))
    Optimization(z; objective, constraints = (;))

Optimisation problem: minimise the [`Functional`](@ref) `objective` over the variables `z` behind it (given or not;
when given they are checked) subject to the bounds of `z` and the named `constraints`, for example
`constraints = (volume = Vol, tip = Constraint(Tip; max = 0.25))`: a functional alone is `F ≤ 0` (as
[`Volume`](@ref HyperShape.Functionals.Volume)), a [`Constraint`](@ref HyperShape.Functionals.Constraint) gives bounds
`max`, `min` or `equal` in the units of the functional. The names go to the history
(`history.constraints.volume`), the banner and the checkpoint. Constraints may read states.

The optimiser treats some constraints itself (OC and SiMPL: one inequality that solves no state, typically the volume;
see `handles`); the others need the augmented Lagrangian, which comes in the next version: until then `solve!` stops
with an error that names them.

A constraint built as linear (`Volume` over identity or filter maps, or a `Functional(…; linear = true)`) is checked
to be linear when the optimisation starts, and costs nothing in the loop (its vector is computed once). A nonlinear
one is linearised at every step by the optimiser, which then corrects it with evaluations of the true constraint
(one filter solve each, no forward solve): see `OC` and `SiMPL`.

The problem owns the context `ctx` ([`OptContext`](@ref)) that reaches the solve functions of the user.
"""
struct Optimization{Z<:OptVariables,J<:AbstractFunctional,C<:NamedTuple}
  z::Z
  objective::J
  constraints::C
  ctx::OptContext
  Optimization(::_Validated, z, J, cs) = new{typeof(z),typeof(J),typeof(cs)}(z, J, cs, OptContext(:optimization))
end

Optimization(z::OptVariables; objective, constraints = (;)) = _optimization(z, objective, constraints)
Optimization(; objective, constraints = (;)) = _optimization(nothing, objective, constraints)

function _optimization(z, objective, constraints)
  objective isa AbstractFunctional || throw(ArgumentError("Optimization: the objective must be a Functional"))
  objective isa Functional && objective.scale_mode === :bound &&
    throw(ArgumentError("Optimization: scale = :bound is only for constraints (it takes the scale from their bounds)"))
  zJ = _optvariables(objective)
  z === nothing && (z = zJ)
  zJ === z || throw(ArgumentError("Optimization: the objective is not built on these variables z"))
  constraints isa Functional &&
    throw(ArgumentError("Optimization: name the constraints, for example constraints = (volume = Vol,)"))
  constraints isa NamedTuple ||
    throw(ArgumentError("Optimization: constraints must be a named tuple, for example (volume = Vol,)"))
  for (n, c) in pairs(constraints)
    n === :objective && throw(ArgumentError("Optimization: objective is not a valid name for a constraint"))
    c isa Union{AbstractFunctional,Constraint} ||
      throw(ArgumentError("Optimization: the constraint $n must be a Functional or a Constraint"))
    _optvariables(c) === z || throw(ArgumentError("Optimization: the constraint $n is not built on these variables z"))
  end
  cs = map(Constraint, constraints)
  for (n, c) in pairs(cs)
    c.F isa Functional && c.F.scale_mode === :bound && isnan(c.F.scale[]) &&
      throw(ArgumentError("Optimization: the constraint $n has scale = :bound and no finite bound"))
  end
  Optimization(_Validated(), z, objective, cs)
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
objective functional at the requested design; each constraint `i` is `gᵢ = (Fᵢ − rᵢ)/sᵢ` with the bounds of its
[`Constraint`](@ref HyperShape.Functionals.Constraint): a linear one gives `(a, V)` computed once at the start
(`linear_constraint`), a nonlinear one is evaluated at the requested design (`constraint`, `constraint_gradient!`; a
design-only one solves no state); `weighted_gradient!` differentiates `w₀ f + Σ wᵢ gᵢ` in one reverse sweep;
`inner`/`riesz!` use the metric of the method.

- It is the only writer of the variables `z`; a design equal to the current one is not set again, so the value cache
  of the functional avoids repeated forward solves.
- It keeps a copy of the design, value and derivative of its last `gradient!`: after a rejected step, the optimiser
  asks again at that design and gets them without solves.
- A gradient that is not finite throws [`OptimizationFailure`](@ref).
"""
mutable struct FunctionalOracle{J,C<:Tuple,Z,M,V<:AbstractVector{Float64}} <: AbstractOracle
  J::J
  cons::C                       # the constraints (Constraint objects)
  z::Z
  ctx::OptContext
  zk::V                         # the loop's current design (shared with the loop)
  lb::V
  ub::V
  a::Vector{Union{Nothing,V}}   # vector of each linear constraint (set at the start), nothing otherwise
  V::Vector{Float64}
  metric::M                     # nothing (Euclidean) or the diagonal metric
  snap_z::V
  snap_dF::V
  snap_f::Float64
  snap_valid::Bool
end

function FunctionalOracle(problem::Optimization, metric)
  z = problem.z
  lb, ub = get_bounds(z)
  m = _metric_vector(metric, z)
  m === nothing || all(>(0), m) || throw(ArgumentError("DensityMethod: the metric has non-positive entries"))
  v = get_values(z)
  cons = Tuple(values(problem.constraints))
  FunctionalOracle(problem.objective, cons, z, problem.ctx, copy(v), copyto!(similar(v), lb), copyto!(similar(v), ub),
                   Union{Nothing,typeof(v)}[nothing for _ in cons], zeros(length(cons)), m,
                   zero(v), zero(v), NaN, false)
end

_invalidate_snapshot!(o::FunctionalOracle) = (o.snap_valid = false; o)

function _set_design!(o::FunctionalOracle, v)
  get_values(o.z) == v || set_values!(o.z, v)
  nothing
end

# Optimisers may hand back the current design through their own variables (SiMPL: a sigmoid of a logit), equal up to
# rounding: such a design is taken as the current one, so it is neither solved again nor treated as a trial.
function _near(v::Vector{Float64}, w::Vector{Float64})
  @inbounds for i in eachindex(v, w)
    abs(v[i] - w[i]) <= 1e-10 * max(abs(w[i]), 1.0) || return false   # SiMPL keeps 1e-12 away from the bounds
  end
  true
end
_near(v, w) = mapreduce((a, b) -> abs(a - b) <= 1e-10 * max(abs(b), 1.0), &, v, w; init = true)
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
    throw(OptimizationFailure("the gradient of the objective is not finite (did the adjoint solve give a finite solution?)"))
  copyto!(o.snap_z, v)
  copyto!(o.snap_dF, dF)
  o.snap_f = functional_value(o.J, o.ctx)       # cached: same design
  o.snap_valid = true
  dF
end

bounds(o::FunctionalOracle) = (o.lb, o.ub)
num_constraints(o::FunctionalOracle) = length(o.cons)

# scale of the functional of a constraint, set (:initial) by evaluating it at the loop's design if needed
function _constraint_scale(o::FunctionalOracle, i)
  F = o.cons[i].F
  s = get_scale(F)
  if isnan(s)
    _set_design!(o, o.zk)
    functional_value(F, o.ctx)
    s = get_scale(F)
  end
  s
end

function constraint_info(o::FunctionalOracle, i)
  c = o.cons[i]
  r = _reference(c)
  s = _constraint_scale(o, i)
  ConstraintInfo((c.lo - r) / s, (c.hi - r) / s; linear = is_linear(c.F), design_only = is_design_only(c.F))
end

linear_constraint(o::FunctionalOracle, i) = (a = o.a[i]; a === nothing ? nothing : (a, o.V[i]))
_is_linear(o::FunctionalOracle, i) = is_linear(o.cons[i].F)

# the value of constraint i: (F - r)/s; with r = 0 (a functional alone, F ≤ 0) exactly the value of the functional
_constraint_value(o::FunctionalOracle, i, ctx) =
  (c = o.cons[i]; functional_value(c.F, ctx) - _reference(c) / get_scale(c.F))

function constraint(o::FunctionalOracle, i::Integer, v)
  a = o.a[i]
  a === nothing || return dot(a, v) - o.V[i]
  v = _canonical(o, v)
  _set_design!(o, v)
  _constraint_value(o, i, o.ctx)
end

function constraint_gradient!(dG, o::FunctionalOracle, i::Integer, v)
  a = o.a[i]
  a === nothing || return copyto!(dG, a)
  v = _canonical(o, v)
  get_values(o.z) == v ||
    throw(ArgumentError("FunctionalOracle: constraint_gradient! asked at a design different from the last constraint"))
  functional_gradient!(dG, o.cons[i].F, o.ctx)
  all(isfinite, dG) || throw(OptimizationFailure("the gradient of the constraint $i is not finite"))
  dG
end

# ∇(w0 f + Σ wᵢ gᵢ): linear constraints through their vectors, the functionals in one reverse sweep
function weighted_gradient!(dL, o::FunctionalOracle, w0, w, v)
  v = _canonical(o, v)
  get_values(o.z) == v ||
    throw(ArgumentError("FunctionalOracle: weighted_gradient! asked at a design different from the last evaluation"))
  set_context!(o.ctx, :phase, :optimization)
  idx = [i for i in eachindex(w) if o.a[i] === nothing && !iszero(w[i])]
  Fs = (o.J, (o.cons[i].F for i in idx)...)
  functional_gradient!(dL, Fs, vcat(Float64(w0), Float64[w[i] for i in idx]), o.ctx)
  for i in eachindex(w)
    o.a[i] === nothing || iszero(w[i]) || (dL .+= w[i] .* o.a[i])
  end
  all(isfinite, dL) || throw(OptimizationFailure("the gradient of the Lagrangian is not finite"))
  dL
end

inner(o::FunctionalOracle{J,C,Z,Nothing}, x, y) where {J,C<:Tuple,Z} = dot(x, y)
function inner(o::FunctionalOracle{J,C,Z,Vector{Float64}}, x::Vector{Float64}, y::Vector{Float64}) where {J,C<:Tuple,Z}
  m = o.metric
  s = zero(eltype(m))
  @inbounds @simd for i in eachindex(m, x, y)
    s += m[i] * x[i] * y[i]
  end
  s
end
# any other vector type: a lazy broadcast reduced without indexing
inner(o::FunctionalOracle, x, y) = sum(Broadcast.instantiate(Broadcast.broadcasted(*, o.metric, x, y)))

riesz!(g, o::FunctionalOracle{J,C,Z,Nothing}, dF) where {J,C<:Tuple,Z} = copyto!(g, dF)
riesz!(g, o::FunctionalOracle, dF) = (g .= dF ./ o.metric)

# a and V of the linear constraint i, g(z) = aᵀz − V (scaled as the functional), with a check of linearity at four
# designs that do not depend on the starting one (relative tolerance 1e-6, so a filter solved iteratively passes)
function _setup_constraint!(o::FunctionalOracle, i::Integer, ctx)
  z = o.z
  c = o.cons[i]
  z0 = copy(get_values(z))
  set_context!(ctx, :phase, :setup)
  a = functional_gradient!(zeros(length(z0)), c.F, ctx)
  v0 = _constraint_value(o, i, ctx)
  (isfinite(v0) && all(isfinite, a)) || throw(ArgumentError("Optimization: the linear constraint is not finite at the start"))
  V = dot(a, z0) - v0
  w = o.ub .- o.lb
  # oscillating designs and uniform shifts (a filter damps the oscillations but not the shifts)
  probes = (o.lb .+ w .* [0.5 + 0.25 * sin(1.7i) for i in eachindex(z0)],
            o.lb .+ w .* [0.5 + 0.25 * cos(2.3i) for i in eachindex(z0)],
            o.lb .+ 0.25 .* w, o.lb .+ 0.8 .* w)
  for z1 in probes
    set_values!(z, z1)
    v1 = _constraint_value(o, i, ctx)
    pred = dot(a, z1) - V
    tol = 1e-6 * (abs(v0) + abs(v1) + sum(abs.(a .* (z1 .- z0))))
    if !(abs(v1 - pred) <= tol)
      set_values!(z, z0)
      throw(ArgumentError("Optimization: the constraint was declared linear but is not linear in z (value $(v1) " *
        "where the linearisation gives $(pred)); declare it with linear = false"))
    end
  end
  set_values!(z, z0)
  o.a[i] = a
  o.V[i] = V
  o
end

_constraint_values(o::FunctionalOracle, zk) = ntuple(i -> constraint(o, i, zk), length(o.cons))
# the multipliers reported by the optimiser, in the order of the constraints (all native in this version)
_multipliers(o::FunctionalOracle, info) =
  ntuple(i -> i <= length(info.multipliers) ? info.multipliers[i] : NaN, length(o.cons))

Base.show(io::IO, m::DensityMethod) = print(io, "DensityMethod(", nameof(typeof(m.optimizer)), ")")
