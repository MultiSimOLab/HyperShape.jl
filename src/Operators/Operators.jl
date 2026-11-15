"""
    HyperShape.Operators

Operators over HyperFEM computational models.

- [`LinearOperator`](@ref): auxiliary linear solves (filter, reaction–diffusion step, velocity extension). The model
  carries the bilinear form only and HyperShape supplies the right-hand side.
- [`StateOperator`](@ref): forward and adjoint states. The model carries the user's full weak forms.

Both declare their dependencies explicitly (`depends = ...`): objects with a [`version`](@ref) counter.
"""
module Operators

using HyperFEM: HyperFEM

export LinearOperator, StateOperator, is_converged, OptContext

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

For example, `RestartState = ctx.attempt > 1` restarts a nonlinear solve from scratch after a rejected step.
"""
mutable struct OptContext
  iteration::Int
  attempt::Int
  trial::Int
  phase::Symbol
  continuation::NamedTuple
end

const PHASES = (:setup, :check, :optimization, :linesearch)

function OptContext(phase::Symbol = :setup)
  phase in PHASES || throw(ArgumentError("OptContext: phase must be one of $(join(PHASES, ", "))"))
  OptContext(0, 0, 0, phase, NamedTuple())
end

Base.setproperty!(::OptContext, s::Symbol, v) =
  throw(ArgumentError("OptContext is read-only: HyperShape updates it"))

# internal setter
function set_context!(ctx::OptContext, s::Symbol, v)
  s === :phase && !(v in PHASES) && throw(ArgumentError("OptContext: unknown phase $v"))
  setfield!(ctx, s, convert(fieldtype(OptContext, s), v))
  ctx
end

"""
    AbstractOperator

Common supertype of [`LinearOperator`](@ref) and [`StateOperator`](@ref).
"""
abstract type AbstractOperator end

_as_depends(d::Tuple) = d
_as_depends(d) = (d,)

function _check_depends(depends)
  for d in depends
    applicable(version, d) || throw(ArgumentError("depends: $(typeof(d)) does not implement HyperShape.Operators.version"))
  end
  depends
end

# loops over the (heterogeneous) tuple of dependencies, unrolled so they do not allocate
_refresh_all!(::Tuple{}, ctx) = nothing
_refresh_all!(t::Tuple, ctx) = (refresh!(first(t), ctx); _refresh_all!(Base.tail(t), ctx))
_any_changed(::Tuple{}, seen, i) = false
_any_changed(t::Tuple, seen, i) = version(first(t)) != seen[i] || _any_changed(Base.tail(t), seen, i + 1)
_mark_seen!(::Tuple{}, seen, i) = nothing
_mark_seen!(t::Tuple, seen, i) = (seen[i] = version(first(t)); _mark_seen!(Base.tail(t), seen, i + 1))

"""
    version(obj) -> Int

Counter that grows every time `obj` changes. Objects listed in the `depends` of an operator implement it.
"""
function version end

"""
    refresh!(obj, ctx)

Brings a dependency up to date before its version is compared: physical fields are updated, state operators are
solved. Nothing for other objects.
"""
refresh!(obj, ctx) = obj

"""
    LinearOperator(model; depends, solve)

Linear solve with a matrix built by a HyperFEM model (typically a `StaticLinearModel` with the bilinear form alone)
and a right-hand side supplied by HyperShape.

- `model`: the computational model; its matrix and solver are the user's.
- `depends`: the objects the matrix depends on (one object or a tuple; each implements [`version`](@ref)); `()` if
  the matrix never changes. Mandatory.
- `solve`: user function `(model, b, ctx, stale) -> x` that solves with the right-hand side `b` and returns the
  solution vector. `stale` is `true` when something in `depends` changed since the last solve, so the user can
  re-assemble, for example `solve = (m, b, ctx, stale) -> solve!(m, b; Assembly = stale)`. Mandatory.

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

function LinearOperator(model; depends, solve)
  deps = _check_depends(_as_depends(depends))
  LinearOperator(model, solve, deps, Int[version(d) for d in deps], false, 0)
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

"""
    num_solves(op) -> Int

Number of solves done by a [`LinearOperator`](@ref) or a [`StateOperator`](@ref) (failed ones included).
"""
num_solves(op::LinearOperator) = op.nsolves

include("StateOperator.jl")

end
