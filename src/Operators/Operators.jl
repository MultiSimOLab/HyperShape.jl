"""
    HyperShape.Operators

Operators over HyperFEM computational models.

- [`LinearOperator`](@ref): auxiliary linear solves (filter, reaction–diffusion step, velocity extension). The model
  carries the bilinear form only and HyperShape supplies the right-hand side.
- [`StateOperator`](@ref): forward states. The model carries the user's full weak forms; an [`AdjointProblem`](@ref)
  declares how to differentiate through it (the transposed solve and the forms `pᵀ ∂R/∂θ`).

Both declare their dependencies (objects with a [`state_version`](@ref) counter), positionally or with `depends = ...`.
For a `StaticLinearModel` both have a default `solve` ([`default_solve`](@ref)); any other model needs the user's.
"""
module Operators

using HyperFEM: HyperFEM
using Gridap
using Gridap.FESpaces
using Gridap.FESpaces: get_dirichlet_dof_values
using Gridap.MultiField: MultiFieldFESpace
using Gridap.Algebra: residual!
using HyperFEMKernels: KernelOperator, update

export LinearOperator, StateOperator, AdjointProblem, SymmetricState, SelfAdjoint, get_adjoint, is_converged, OptContext, Parameter

"""
    OptContext(phase::Symbol = :setup)

Context of the optimisation, passed to every solve function of the user (`solve = (m, ctx) -> …`) and to the
post-processing. Read-only for the user; HyperShape updates it. Fields:

- `iteration`: design iteration (0 before the loop starts).
- `attempt`: attempt of the current iteration (1, or more after rejected steps).
- `trial`: number of forward evaluations requested in the current iteration.
- `phase`: `:setup` (constructors), `:check` ([`check_consistency`](@ref HyperShape.Checks.check_consistency)),
  `:optimization` (the current design of the loop and its adjoint) or `:linesearch` (a trial design requested by the
  optimiser; with SiMPL the design that is finally accepted is first solved in this phase).
- `continuation`: named tuple with the current values of the continued parameters, for example `(β = 4.0,)`.
- `sensitivity`: `:design` (the derivative with respect to the design, through every map) or `:physical` (set by
  [`ReactionDiffusion`](@ref HyperShape.Methods.ReactionDiffusion) while it computes the sensitivity of its step, when
  the design chain has an interface map: the reverse sweep takes an interface map, `Step()` or `Ersatz()`, as the identity weighted
  by the material, `δ ↦ (w₀ + (1 − w₀) χ) δ`); `:design` everywhere else, also in
  [`check_consistency`](@ref HyperShape.Checks.check_consistency).
- `void_weight`: the weight `w₀` of the void in the `:physical` sensitivity.

For example, `RestartState = ctx.trial > 1` restarts a nonlinear solve from scratch at every evaluation after the
first one of an iteration (a design tried after a rejected step, or a line-search trial).
"""
mutable struct OptContext
  iteration::Int
  attempt::Int
  trial::Int
  phase::Symbol
  continuation::NamedTuple
  sensitivity::Symbol
  void_weight::Float64
end

const PHASES = (:setup, :check, :optimization, :linesearch)

function OptContext(phase::Symbol = :setup)
  phase in PHASES || throw(ArgumentError("OptContext: phase must be one of $(join(PHASES, ", "))"))
  OptContext(0, 0, 0, phase, NamedTuple(), :design, 0.0)
end
# the positional form of the five fields before `sensitivity` and `void_weight`
OptContext(iteration::Integer, attempt::Integer, trial::Integer, phase::Symbol, continuation::NamedTuple) =
  OptContext(iteration, attempt, trial, phase, continuation, :design, 0.0)

Base.setproperty!(::OptContext, s::Symbol, v) =
  throw(ArgumentError("OptContext is read-only: HyperShape updates it"))

# internal setter
function set_context!(ctx::OptContext, s::Symbol, v)
  s === :phase && !(v in PHASES) && throw(ArgumentError("OptContext: unknown phase $v"))
  s === :sensitivity && !(v in (:design, :physical)) && throw(ArgumentError("OptContext: sensitivity must be :design or :physical"))
  setfield!(ctx, s, convert(fieldtype(OptContext, s), v))
  ctx
end

"""
    AbstractOperator

Common supertype of [`LinearOperator`](@ref) and [`StateOperator`](@ref).
"""
abstract type AbstractOperator end

"""
    AbstractResponse

Supertype of the operators that a functional can be differentiated through: a forward state with its adjoint
([`StateOperator`](@ref) with an [`AdjointProblem`](@ref)). Declared for the reverse sweep of chains of states,
trajectories and eigenproblems (later versions); its contract is not fixed yet.
"""
abstract type AbstractResponse <: AbstractOperator end

_as_depends(d::Tuple) = d
_as_depends(d) = (d,)

# dependencies given positionally or with `depends =`, never both
struct _Unset end
function _dependencies(who, positional, keyword)
  positional isa _Unset || keyword isa _Unset ||
    throw(ArgumentError("$who: give the dependencies once, positionally or with depends ="))
  positional isa _Unset ? keyword : positional
end

"""
    default_solve(::Type{<:AbstractOperator}, model)

Solve function used when the user gives none. Only a HyperFEM `StaticLinearModel` has one, and it calls the model's
own solver (HyperShape never chooses one):

- [`StateOperator`](@ref): `(m, ctx) -> solve!(m; Assembly = true)`. The operator is solved only when a dependency
  changed, so the forms are always re-assembled (`Assembly = false` would also reuse the previous right-hand side).
- [`LinearOperator`](@ref): `(m, b, ctx, stale) -> solve!(m, b; Assembly = stale)`: the matrix is re-assembled and
  factorised again only when a dependency changed.

Any other model (nonlinear, staggered, transient, or a `StaticLinearModel` with several right-hand sides) throws: the load stepping and the restart are modelling decisions,
so pass `solve = (m, ctx) -> solve!(m; stepping = …, RestartState = …)`. To use another solver (PETSc, an iterative
solver with a preconditioner…) build the model with it, or pass your own `solve`.
"""
function default_solve end

_solve_state_linear(m, ctx) = HyperFEM.solve!(m; Assembly = true)
_solve_rhs_linear(m, b, ctx, stale) = HyperFEM.solve!(m, b; Assembly = stale)

default_solve(::Type{<:AbstractOperator}, model) =
  throw(ArgumentError("no default solve for a $(nameof(typeof(model))): only a StaticLinearModel with one " *
                      "right-hand side has one. For a nonlinear model the load stepping and the restart are modelling " *
                      "decisions: pass solve = (m, ctx) -> solve!(m; stepping = …, RestartState = …) (or your own solve)"))

function _check_depends(depends)
  for d in depends
    applicable(state_version, d) || throw(ArgumentError("depends: $(typeof(d)) does not implement HyperShape.state_version"))
  end
  depends
end

# loops over the (heterogeneous) tuple of dependencies, unrolled so they do not allocate
_refresh_all!(::Tuple{}, ctx) = nothing
_refresh_all!(t::Tuple, ctx) = (refresh!(first(t), ctx); _refresh_all!(Base.tail(t), ctx))
_any_changed(::Tuple{}, seen, i) = false
_any_changed(t::Tuple, seen, i) = state_version(first(t)) != seen[i] || _any_changed(Base.tail(t), seen, i + 1)
_mark_seen!(::Tuple{}, seen, i) = nothing
_mark_seen!(t::Tuple, seen, i) = (seen[i] = state_version(first(t)); _mark_seen!(Base.tail(t), seen, i + 1))

"""
    state_version(obj) -> Int

Counter that grows every time `obj` changes. Objects listed in the `depends` of an operator implement it.
"""
function state_version end

"""
    refresh!(obj, ctx)

Brings a dependency up to date before its version is compared: physical fields are updated, state operators are
solved. Nothing for other objects.
"""
refresh!(obj, ctx) = obj

"""
    Parameter(value)

A number of the user that may change during the optimisation: the penalty of a material interpolation, a load
factor, a weight. `p[]` reads it and `p[] = v` changes it (its [`state_version`](@ref) grows only if the value
changes). Read it inside the weak forms (`E(ρ) = Emin + ρ^p[] * (1 - Emin)`) and give it as a dependency of the
operators whose models read it (`StateOperator(model, (θ, p))`), so they solve again when it changes; inside a map,
give it in the `parameters` of a `PointwiseMap`. Continue it in stages with `Continuation(p = p => [1, 2, 3]; every)`.
A functional that reads it directly (not through a state or θ) lists it in its `parameters`. A parameter read by a
model but not declared as a dependency is not detected by `check_consistency`: after it changes, the operator would
keep its old solution.
"""
mutable struct Parameter
  value::Float64
  version::Int
  function Parameter(v::Real)
    isfinite(v) || throw(ArgumentError("Parameter: the value must be finite"))
    new(Float64(v), 0)
  end
end

Base.getindex(p::Parameter) = p.value
function Base.setindex!(p::Parameter, v::Real)
  isfinite(v) || throw(ArgumentError("Parameter: the value must be finite"))
  if v != p.value
    p.value = Float64(v)
    p.version += 1
  end
  p
end
state_version(p::Parameter) = p.version
Base.show(io::IO, p::Parameter) = print(io, "Parameter(", p.value, ")")

"""
    LinearOperator(model[, deps]; depends = (), solve = default_solve(LinearOperator, model))

Linear solve with a matrix built by a HyperFEM model (typically a `StaticLinearModel` with the bilinear form alone)
and a right-hand side supplied by HyperShape.

- `model`: the computational model; its matrix and solver are the user's.
- `deps` or `depends`: the objects the matrix depends on (one object or a tuple; each implements
  [`state_version`](@ref)), given once. Default `()`: the matrix never changes. A dependency captured by the bilinear form
  and not declared gives stale results; [`check_consistency`](@ref HyperShape.Checks.check_consistency) detects it.
- `solve`: function `(model, b, ctx, stale) -> x` that solves with the right-hand side `b` and returns the solution
  vector. `stale` is `true` when something in the dependencies changed since the last solve, so it can re-assemble.
  Default for a `StaticLinearModel`: `solve!(m, b; Assembly = stale)` ([`default_solve`](@ref)); mandatory for any
  other model.

The matrix must be the one the model assembled at construction (no dependency has changed yet). The returned vector
is copied, so it may be a buffer of the model. [`invalidate!`](@ref) makes the next solve stale whatever the
dependencies (the consistency checks use it).
"""
mutable struct LinearOperator{M,F,D} <: AbstractOperator
  model::M
  solve::F
  depends::D
  seen::Vector{Int}
  force_stale::Bool   # the next solve is stale whatever the dependencies (set by the consistency checks)
  nsolves::Int        # number of solves done
end

function LinearOperator(model, deps = _Unset(); depends = _Unset(), solve = default_solve(LinearOperator, model))
  d = _dependencies("LinearOperator", deps, depends)
  deps = _check_depends(_as_depends(d isa _Unset ? () : d))
  LinearOperator(model, solve, deps, Int[state_version(d) for d in deps], false, 0)
end

get_model(op::LinearOperator) = op.model

"""
    solve_linear!(x, op::LinearOperator, b, ctx) -> x

Solves with the right-hand side `b` through the user's `solve` and copies the solution into `x`.
"""
function solve_linear!(x, op::LinearOperator, b, ctx)
  _refresh_all!(op.depends, ctx)
  stale = op.force_stale || _any_changed(op.depends, op.seen, 1)
  y = op.solve(op.model, b, ctx, stale)
  length(y) == length(x) ||
    throw(DimensionMismatch("the solve function of the LinearOperator returned $(length(y)) values, expected $(length(x))"))
  # the dependencies count as seen only after a successful solve: if the solve throws, the next call is stale again
  _mark_seen!(op.depends, op.seen, 1)
  op.force_stale = false
  op.nsolves += 1
  copyto!(x, y)
end

"""
    invalidate!(obj) -> obj

Forgets what `obj` computed, so that it is computed again on next use: the next solve of a
[`LinearOperator`](@ref) is stale (the user re-assembles), a [`StateOperator`](@ref) is solved again (until then
[`is_converged`](@ref) is `false`), physical fields recompute their maps and a functional its value. Used by
[`check_consistency`](@ref HyperShape.Checks.check_consistency); physical fields and functionals extend it.
"""
function invalidate! end

invalidate!(op::LinearOperator) = (op.force_stale = true; op)

Base.show(io::IO, op::LinearOperator) =
  print(io, "LinearOperator(", nameof(typeof(op.model)), ", ", op.nsolves, " solves)")

"""
    num_solves(op) -> Int

Number of solves done by a [`LinearOperator`](@ref) or a [`StateOperator`](@ref) (failed ones included).
"""
num_solves(op::LinearOperator) = op.nsolves

include("Assembly.jl")
include("StateOperator.jl")
include("Adjoint.jl")

end
