# Problem, metric and the oracle that connects the functionals with the optimisers.

_optvariables(F::AbstractFunctional) = get_optvariables(get_designfields(physical_fields(F)))
_optvariables(c::Constraint) = _optvariables(c.F)

struct _Validated end   # only the keyword constructors build problems

# the factories of a problem and the model they are called with
struct ProblemSetup{M,D,P}
  model::M
  design_setup::D
  physics_setup::P
end

"""
    Optimization(; objective, constraints = (;))
    Optimization(z; objective, constraints = (;))
    Optimization(model; design_setup, physics_setup)

Optimisation problem: minimise the [`Functional`](@ref) `objective` over the variables `z` behind it (given or not;
when given they are checked) subject to the bounds of `z` and the named `constraints`, for example
`constraints = (volume = Vol, tip = Constraint(Tip; max = 0.25))`: a functional alone is `F ≤ 0` (as
[`Volume`](@ref HyperShape.Functionals.Volume)), a [`Constraint`](@ref HyperShape.Functionals.Constraint) gives bounds
`max`, `min` or `equal` in the units of the functional. The names go to the history
(`history.constraints.volume`), the banner and the checkpoint. Constraints may read states.

The optimiser treats some constraints itself (OC and SiMPL: one inequality that solves no state, typically the volume;
see `handles`); the others enter its objective through the augmented Lagrangian of the method
(`DensityMethod(…; lagrangian = AugmentedLagrangian(…))`, used by default and shown in the banner).

A constraint built as linear (`Volume` over identity or filter maps, or a `Functional(…; linear = true)`) is checked
to be linear when the optimisation starts, and costs nothing in the loop (its vector is computed once). A nonlinear
one is linearised at every step by the optimiser, which then corrects it with evaluations of the true constraint
(one filter solve each, no forward solve): see `OC` and `SiMPL`.

The third form builds the problem from two functions of the user (factories), so that `solve!` can build it again
during the run (see `discretization` in [`solve!`](@ref)): `design_setup(model)` returns a named tuple with `z` (the
[`OptVariables`](@ref), on spaces of `model`) and `x` (the [`DesignFields`](@ref) built on `z`); `physics_setup(model,
x)` returns a named tuple with `objective` and optionally `constraints`, `monitors` (as the keyword of `solve!`) and
`output` (the driver of an [`OptPostProcessor`](@ref HyperShape.PostProcess.OptPostProcessor) built without one,
`(post, ctx) -> …`). Both are called once here and again at each rebuild; every name they return is kept in
`problem.design` and `problem.physics` (for example `problem.physics.θ`). What is continued or does not depend on the
mesh (a `TanhProjection` of a `Continuation`, a `Parameter`, the optimiser) is created once, outside the factories,
and used inside them; everything that depends on the mesh (spaces, measures, fields, states, functionals, monitors,
output) is created inside. The factories may be called several times: they should have no other effects. A name
close to one of the contract (`monitor`, `constraint`, `outputs`, `post`, `driver`, `objectives`, `J`) without the
right one is an error. After a run with rebuilds the variable `problem` of the script is the first problem:
the last one is `result.problem` (both share `problem.state[]`).

The problem owns the context `ctx` ([`OptContext`](@ref)) that reaches the solve functions of the user, and
`problem.state[]`, the last [`OptState`](@ref) captured by `solve!` (at the end of a run, after a failure or an
interruption; a `solve!` that throws before its first iteration leaves the previous one): pass it as `resume` to
continue without files.
"""
struct Optimization{Z<:OptVariables,J<:AbstractFunctional,C<:NamedTuple}
  z::Z
  objective::J
  constraints::C
  ctx::OptContext
  state::Base.RefValue{Any}     # the last OptState captured by solve! (nothing before); shared by rebuilt problems
  setup::Union{Nothing,ProblemSetup}   # the factories (nothing for a problem built from its functionals)
  design::NamedTuple            # what design_setup returned ((; z) without factories); read only when a run is set up
  physics::NamedTuple           # what physics_setup returned ((; objective, constraints) without factories)
  function Optimization(::_Validated, z, J, cs; ctx = OptContext(:optimization), state = Ref{Any}(nothing), setup = nothing,
                        design = (; z), physics = (; objective = J, constraints = cs))
    new{typeof(z),typeof(J),typeof(cs)}(z, J, cs, ctx, state, setup, design, physics)
  end
end

Optimization(z::OptVariables; objective, constraints = (;)) = _optimization(z, objective, constraints)
Optimization(; objective, constraints = (;)) = _optimization(nothing, objective, constraints)

function Optimization(model; design_setup = nothing, physics_setup = nothing, kwargs...)
  (design_setup === nothing || physics_setup === nothing || !isempty(kwargs)) &&
    throw(ArgumentError("Optimization: give the variables z and the functionals, Optimization(z; objective, constraints), " *
                        "or a model and the two factories, Optimization(model; design_setup, physics_setup)"))
  _from_setup(ProblemSetup(model, design_setup, physics_setup), OptContext(:optimization), Ref{Any}(nothing), nothing)
end

"""
    HyperShape.Methods.rebuild(problem; what = :design, model = problem.setup.model) -> Optimization

Builds the problem again with its factories (a problem built with `Optimization(model; design_setup, physics_setup)`):
both (`what = :design`) or only `physics_setup`, with the design of `problem` (`what = :physics`), on `model` (the
same one by default; another model is for the discretisations that change the mesh, later). The new problem
shares the context and `state` of `problem`; the values of `z` are not copied (`solve!` restores them from the state
at a rebuild). Used by the discretisations of [`solve!`](@ref) (`RebuildCheck`).
"""
function rebuild(problem::Optimization; what::Symbol = :design, model = nothing)
  problem.setup === nothing &&
    throw(ArgumentError("rebuild: the problem has no factories; build it with Optimization(model; design_setup, physics_setup)"))
  what in (:design, :physics) || throw(ArgumentError("rebuild: what must be :design or :physics"))
  setup = problem.setup
  if model !== nothing && model !== setup.model
    what === :physics && throw(ArgumentError("rebuild: a new model needs the design built again (what = :design)"))
    setup = ProblemSetup(model, setup.design_setup, setup.physics_setup)
  end
  _from_setup(setup, problem.ctx, problem.state, what === :physics ? problem.design : nothing)
end

"""
    check_consistency(problem::Optimization; kwargs...)

[`check_consistency`](@ref HyperShape.Checks.check_consistency) of the objective and the constraints of `problem`, by
name (`objective` and the names of the constraints).
"""
check_consistency(p::Optimization; kwargs...) =
  check_consistency(NamedTuple{(:objective, keys(p.constraints)...)}((p.objective, (c.F for c in p.constraints)...)); kwargs...)

# names of the contract and the near misses that are taken for them
const _DESIGN_NAMES = (:z, :x)
const _PHYSICS_NAMES = (:objective, :constraints, :monitors, :output)
const _NEAR_NAMES = Dict(:monitor => :monitors, :constraint => :constraints, :outputs => :output, :post => :output,
                         :driver => :output, :objectives => :objective, :J => :objective)

function _check_names(r, who)
  for n in keys(r)
    m = get(_NEAR_NAMES, n, nothing)
    m === nothing || haskey(r, m) ||
      throw(ArgumentError("Optimization: $who returns $n; did you mean $m?"))
  end
end

function _from_setup(setup::ProblemSetup, ctx, state, design)
  model = setup.model
  if design === nothing
    design = setup.design_setup(model)
    (design isa NamedTuple && haskey(design, :z) && haskey(design, :x)) ||
      throw(ArgumentError("Optimization: design_setup(model) must return a named tuple with z (OptVariables) and x " *
                          "(DesignFields built on z), for example (; z, x)"))
    _check_names(design, "design_setup")
    design.z isa OptVariables || throw(ArgumentError("Optimization: the z returned by design_setup must be an OptVariables"))
    (design.x isa DesignFields && get_optvariables(design.x) === design.z) ||
      throw(ArgumentError("Optimization: the x returned by design_setup must be the DesignFields built on its z"))
    for (n, b) in zip(design.z.names, design.z.blocks)
      _on_model(b.space, model) ||
        throw(ArgumentError("Optimization: the design space of $n is not on the model given to design_setup; build it " *
                            "from the model the factory receives"))
    end
  end
  physics = setup.physics_setup(model, design.x)
  (physics isa NamedTuple && haskey(physics, :objective)) ||
    throw(ArgumentError("Optimization: physics_setup(model, x) must return a named tuple with objective (and optionally " *
                        "constraints, monitors, output), for example (; objective = C, constraints = (volume = Vol,))"))
  _check_names(physics, "physics_setup")
  F = physics.objective
  (F isa AbstractFunctional && get_designfields(physical_fields(F)) === design.x) ||
    throw(ArgumentError("Optimization: the objective returned by physics_setup is not built on the x it received"))
  haskey(physics, :output) && !hasmethod(physics.output, Tuple{OptPostProcessor{_SetupDriver},OptContext}) &&
    throw(ArgumentError("Optimization: the output returned by physics_setup must be a function (post, ctx) -> …"))
  p = _optimization(design.z, F, get(physics, :constraints, (;)))
  Optimization(_Validated(), p.z, p.objective, p.constraints; ctx, state, setup, design, physics)
end

# the space is built on `model` (the background model of its triangulation; a design space without one is accepted)
function _on_model(space, model)
  m = try
    get_background_model(get_triangulation(space))
  catch
    return true
  end
  m === model
end

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

    LumpedMass()

The same on the triangulation of each design space, with a quadrature of degree 2 (exact for designs of order up to
2): no measure to give, and it follows the design spaces if the mesh changes.
"""
struct LumpedMass{M}
  measure::M
end
LumpedMass() = LumpedMass(nothing)

const Metric = Union{Euclidean,LumpedMass}

"""
    AbstractMethod

Supertype of the methods (`DensityMethod`, `ReactionDiffusion`; level set later). All methods share the loop of
[`solve!`](@ref) (evaluation, acceptance, continuation, augmented Lagrangian, stopping, checkpoint, output); a method
only gives its step and what the step sees of the problem, through four functions (public, not exported):

- `stepper(m)`: the step, an `Optimizer` (`init`, `step!`, `reset!`, `shrink!`, `memory`, `restore_memory!`,
  `objective_changed!`; see `HyperShape.Optimizers`); the same object at every call (a continuation compares it);
- `build_oracle(m, problem)`: the oracle the step sees (objective, gradient, constraints, bounds, metric); for now it
  must be a `HyperShape.Methods.FunctionalOracle` (internal: its interface may change), built for example with
  `FunctionalOracle(problem, metric)`;
- `augmented_lagrangian(m)`: the [`AugmentedLagrangian`](@ref HyperShape.Optimizers.AugmentedLagrangian) of the constraints
  the step does not treat;
- `check_problem(m, problem)`: errors when the run starts (default: none).

`method_label(m)` (default: the name of the type) and `metric_label(m)` (default: none) go to the banner, with the
lines of `banner_lines(m, problem)` (default: none; a vector of `(text, colour)`). `feasible_start(m)` (default
`true`): whether a starting design that violates the native constraint is scaled to meet it before the first step
(`ReactionDiffusion`: `false`, its step lowers the volume).
"""
abstract type AbstractMethod end

function stepper end
function build_oracle end
function augmented_lagrangian end
check_problem(::AbstractMethod, problem) = nothing
method_label(m::AbstractMethod) = string(nameof(typeof(m)))
metric_label(::AbstractMethod) = nothing
banner_lines(::AbstractMethod, problem) = Tuple{String,Symbol}[]
feasible_start(::AbstractMethod) = true

"""
    DensityMethod(optimizer; metric = default_metric(optimizer), lagrangian = AugmentedLagrangian())

Density method (SIMP, RAMP or any interpolation written by the user in the weak forms): the step is `optimizer` (any
`Optimizer`: `OC(…)`, `SiMPL(…)`). `metric`: [`LumpedMass`](@ref)`(dΩ)`, [`Euclidean`](@ref)`()`, a vector of
positive weights (one per entry of z: `inner(x, y) = Σ mᵢ xᵢ yᵢ`), an object of the user with a method
[`metric_weights`](@ref)`(m, z) -> Vector` (called once at the start of `solve!` and again after each rebuild of the problem), or a named tuple with one of them
per block of z. SiMPL uses
it in its step and its optimality measure, and needs it (no default). OC uses it only for the stationarity measure
(see [`StopCriteria`](@ref)); its default is `Euclidean()`, which makes that measure depend on the mesh.

`lagrangian`: how the constraints the optimiser does not treat itself are handled, an
[`AugmentedLagrangian`](@ref HyperShape.Optimizers.AugmentedLagrangian) (default `AugmentedLagrangian()`:
[`Bertsekas`](@ref HyperShape.Optimizers.Bertsekas) with a fixed penalty `μ = 1`, multipliers updated after every
accepted iteration, the first constraints the optimiser handles kept native). It is used only when there is such a
constraint, or when its `native` names fewer than the optimiser would take.
"""
struct DensityMethod{O<:Optimizer,M,L<:AugmentedLagrangian} <: AbstractMethod
  optimizer::O
  metric::M
  default_metric::Bool   # the metric was not given (shown in the banner)
  lagrangian::L
  default_lagrangian::Bool
  function DensityMethod(optimizer::Optimizer; metric = nothing, lagrangian = nothing)
    lgiven = lagrangian !== nothing
    lgiven || (lagrangian = AugmentedLagrangian())
    lagrangian isa AugmentedLagrangian ||
      throw(ArgumentError("DensityMethod: lagrangian must be an AugmentedLagrangian, for example AugmentedLagrangian(Bertsekas(μ0 = 0.1))"))
    given = metric !== nothing
    given || (metric = default_metric(optimizer))
    ok = _valid_metric(metric) || (metric isa NamedTuple && all(_valid_metric, values(metric)))
    ok || throw(ArgumentError("DensityMethod: metric must be LumpedMass(), LumpedMass(dΩ), Euclidean(), a vector of positive weights, " *
                              "an object with a metric_weights(m, z) method, or a named tuple of them per block"))
    new{typeof(optimizer),typeof(metric),typeof(lagrangian)}(optimizer, metric, !given, lagrangian, !lgiven)
  end
end

stepper(m::DensityMethod) = m.optimizer
build_oracle(m::DensityMethod, problem) = FunctionalOracle(problem, m.metric)
augmented_lagrangian(m::DensityMethod) = m.lagrangian
function check_problem(::DensityMethod, problem)
  lb, ub = get_bounds(problem.z)
  (all(isfinite, lb) && all(isfinite, ub)) || throw(ArgumentError("DensityMethod: the bounds of z must be finite"))
  # the step reads the derivative with respect to z, through the maps x → θ
  for (n, F) in pairs((objective = problem.objective, map(c -> c.F, problem.constraints)...))
    θ = physical_fields(F)
    θ isa PhysicalFields && (!is_differentiable(θ) || has_ersatz(θ)) &&
      throw(ArgumentError("DensityMethod: the $(n === :objective ? "objective" : "constraint $n") reads a physical field " *
                          "through an interface map (Step(), Ersatz()); interface maps are for interface methods " *
                          "(reaction–diffusion), a density method interpolates the density itself (SIMP, RAMP)"))
  end
  nothing
end
method_label(m::DensityMethod) = "DensityMethod($(nameof(typeof(m.optimizer))))"

"""
    default_metric(optimizer)

Metric of [`DensityMethod`](@ref) when none is given: `Euclidean()` for OC (which only uses it for the stationarity measure); an error for any other
optimiser, which must be given one.
"""
default_metric(::OC) = Euclidean()
default_metric(opt::Optimizer) =
  throw(ArgumentError("DensityMethod: $(nameof(typeof(opt))) needs a metric: DensityMethod(opt; metric = LumpedMass()) or Euclidean()"))

# an object of the user is checked when the weights are asked for (its metric_weights may restrict the type of z)
_valid_metric(m) = !(m isa Union{Nothing,Symbol,AbstractString,Number,Tuple,AbstractVector}) ||
                   m isa AbstractVector{<:Real}

"""
    metric_weights(m, z) -> AbstractVector

Weights of a diagonal metric of the user, one per entry of the variables `z` (or of the block, in a named tuple of
metrics: then `z` is the space of that block). Called once at the start of [`solve!`](@ref) and again after each rebuild of the problem; they must be positive.
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
_block_metric(::LumpedMass{Nothing}, V) = (dΩ = Measure(get_triangulation(V), 2); collect(assemble_vector(v -> ∫(v)dΩ, V)))

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
mutable struct FunctionalOracle{J,C<:Tuple,Z,M,V<:AbstractVector{Float64},FS<:Tuple} <: AbstractOracle
  J::J
  cons::C                       # the constraints (Constraint objects)
  Fs::FS                        # (objective, functional of each constraint): one tuple type for weighted_gradient!
  wbuf::Vector{Float64}         # its weights
  infos::Vector{Union{Nothing,ConstraintInfo}}   # constraint_info, kept once the scale is set
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
  Fs = (problem.objective, map(c -> c.F, cons)...)
  FunctionalOracle(problem.objective, cons, Fs, zeros(length(Fs)), Union{Nothing,ConstraintInfo}[nothing for _ in cons], z,
                   problem.ctx, copy(v), copyto!(similar(v), lb), copyto!(similar(v), ub),
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
    isfinite(s) || throw(ArgumentError("Optimization: the constraint $i is not finite at the starting design (did its " *
                                       "state converge?); its scale = :initial needs a finite first value"))
  end
  s
end

function constraint_info(o::FunctionalOracle, i)
  info = o.infos[i]
  info === nothing || return info
  c = o.cons[i]
  r = _reference(c)
  s = _constraint_scale(o, i)
  o.infos[i] = ConstraintInfo((c.lo - r) / s, (c.hi - r) / s; linear = is_linear(c.F), design_only = is_design_only(c.F))
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
  length(w) == length(o.cons) ||
    throw(DimensionMismatch("weighted_gradient!: $(length(w)) weights for $(length(o.cons)) constraints"))
  v = _canonical(o, v)
  _set_design!(o, v)                     # a design served from the snapshot may not be the one in z
  set_context!(o.ctx, :phase, :optimization)
  wb = o.wbuf
  wb[1] = w0
  for i in eachindex(w)
    wb[i + 1] = o.a[i] === nothing ? w[i] : 0.0      # linear constraints through their vectors
  end
  functional_gradient!(dL, o.Fs, wb, o.ctx)
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

_constraint_values(o::FunctionalOracle{J,C}, zk) where {J,C} = ntuple(i -> constraint(o, i, zk), Val(fieldcount(C)))
# the multipliers reported by the optimiser, in the order of the constraints (all native in this version)
_multipliers(o::FunctionalOracle{J,C}, info) where {J,C} =
  ntuple(i -> i <= length(info.multipliers) ? info.multipliers[i] : NaN, Val(fieldcount(C)))

Base.show(io::IO, m::DensityMethod) = print(io, "DensityMethod(", nameof(typeof(m.optimizer)), ")")
