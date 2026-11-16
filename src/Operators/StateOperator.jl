"""
    StateOperator(model; depends, solve)

Forward or adjoint state: a HyperFEM computational model built by the user (weak forms and solver are the user's)
that is solved again only when one of its dependencies changed.

- `depends`: the objects the state depends on (one object or a tuple), typically `θ` for a forward state and
  `(θ, S)` for its adjoint. Each implements [`version`](@ref); physical fields and other state operators are
  brought up to date first. Mandatory.
- `solve`: user function `(model, ctx) -> result` that solves the model, for example
  `(m, ctx) -> solve!(m; stepping = …)` or `(m, ctx) -> solve!(m; Assembly = true)`. The result is either the
  solution vector (taken as converged) or a tuple whose last element is the convergence flag: a `Bool`, or `1`/`0`
  as the `flagconv` of `(x, flagconv)` returned by HyperFEM's nonlinear models. Mandatory.

`get_state(S)` is the state field of the model, created once, so the forms of other models (the adjoint) can
capture it; it always holds the model's current solution vector.

Failures do not throw. When the solve does not converge, or a state this one depends on did not converge (then
this one is not solved), [`apply!`](@ref) returns `false`, [`is_converged`](@ref) is `false` and the functionals
that need the state return `NaN`. After a failure `get_state(S)` holds whatever the model left (for HyperFEM's
nonlinear models, the last converged load increment), not a solution. A failed solve is not repeated until a
dependency changes; the next solve starts from what the model holds, as decided by the user's `solve`
(`RestartState` in HyperFEM).
"""
mutable struct StateOperator{M,F,D,U} <: AbstractOperator
  model::M
  solve::F
  depends::D
  seen::Vector{Int}
  state::U
  version::Int
  converged::Bool
  solved::Bool
end

function StateOperator(model; depends, solve)
  deps = _check_depends(_as_depends(depends))
  StateOperator(model, solve, deps, fill(-1, length(deps)), HyperFEM.get_state(model), 0, false, false)
end

version(S::StateOperator) = S.version
HyperFEM.get_state(S::StateOperator) = S.state
get_model(S::StateOperator) = S.model

"""
    is_converged(S::StateOperator) -> Bool

Whether the current state of `S` is a converged solution (`false` before the first solve).
"""
is_converged(S::StateOperator) = S.solved && S.converged

function _converged(r::Tuple)
  flag = last(r)
  flag isa Bool && return flag
  flag isa Integer && (flag == 0 || flag == 1) && return flag == 1   # HyperFEM's flagconv: 1 converged, 0 failed
  throw(ArgumentError("StateOperator: the solve function returned a tuple whose last element is $(repr(flag)); expected the convergence flag (a Bool, or 1/0 as HyperFEM's flagconv)"))
end
_converged(r) = true

# a dependency that is a state must have converged
_dependencies_converged(::Tuple{}) = true
_dependencies_converged(t::Tuple) = _dependency_converged(first(t)) && _dependencies_converged(Base.tail(t))
_dependency_converged(d) = true
_dependency_converged(d::StateOperator) = is_converged(d)

"""
    apply!(S::StateOperator, ctx) -> Bool

Brings the dependencies up to date and solves the model if one of them changed (or if it was never solved).
Returns whether the current state converged. A failed solve is not repeated until a dependency changes; if a state
this one depends on did not converge, this one is not solved and counts as not converged.
"""
function apply!(S::StateOperator, ctx)
  _refresh_all!(S.depends, ctx)
  (S.solved && !_any_changed(S.depends, S.seen, 1)) && return S.converged
  S.converged = _dependencies_converged(S.depends) ? _converged(S.solve(S.model, ctx)) : false
  _mark_seen!(S.depends, S.seen, 1)
  S.solved = true
  S.version += 1
  S.converged
end

refresh!(S::StateOperator, ctx) = (apply!(S, ctx); S)

invalidate!(S::StateOperator) = (S.solved = false; S)
num_solves(S::StateOperator) = S.version
