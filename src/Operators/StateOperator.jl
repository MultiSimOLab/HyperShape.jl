"""
    StateOperator(model; depends, solve)

Forward or adjoint state: a HyperFEM computational model built by the user (weak forms and solver are the user's)
that is solved again only when one of its dependencies changed.

- `depends`: tuple of the objects the state depends on, typically `(θ,)` for a forward state and `(θ, S)` for its
  adjoint. Each implements [`version`](@ref); physical fields and other state operators are brought up to date
  first. Mandatory.
- `solve`: user function `(model, ctx) -> result` that solves the model, for example
  `(m, ctx) -> solve!(m; stepping = …)` or `(m, ctx) -> solve!(m; Assembly = true)`. The result is either the
  solution vector (taken as converged) or a tuple whose last element is the convergence flag (as `(x, flagconv)`
  from HyperFEM's nonlinear models). Mandatory.

`get_state(S)` is the state field of the model: a fixed object, so the forms of other models (the adjoint) can
capture it. A failed solve does not throw: [`apply!`](@ref) returns `false` and the functionals that need the
state return `NaN`.
"""
mutable struct StateOperator{M,F,D,U}
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
  depends isa Tuple || throw(ArgumentError("depends must be a tuple, for example (θ,)"))
  for d in depends
    applicable(version, d) || throw(ArgumentError("depends: $(typeof(d)) does not implement HyperShape.Operators.version"))
  end
  StateOperator(model, solve, depends, fill(-1, length(depends)), HyperFEM.get_state(model), 0, false, false)
end

version(S::StateOperator) = S.version
HyperFEM.get_state(S::StateOperator) = S.state
get_model(S::StateOperator) = S.model
is_converged(S::StateOperator) = S.solved && S.converged

_converged(r::Tuple) = Bool(last(r))
_converged(r) = true

"""
    apply!(S::StateOperator, ctx) -> Bool

Brings the dependencies up to date and solves the model if one of them changed (or if it was never solved).
Returns whether the current state converged. A failed solve is not repeated until a dependency changes.
"""
function apply!(S::StateOperator, ctx)
  foreach(d -> refresh!(d, ctx), S.depends)
  changed = !S.solved
  for (i, d) in enumerate(S.depends)
    changed |= version(d) != S.seen[i]
  end
  changed || return S.converged
  r = S.solve(S.model, ctx)
  for (i, d) in enumerate(S.depends)
    S.seen[i] = version(d)
  end
  S.converged = _converged(r)
  S.solved = true
  S.version += 1
  S.converged
end

refresh!(S::StateOperator, ctx) = (apply!(S, ctx); S)
