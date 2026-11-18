"""
    StateOperator(model, deps; solve = default_solve(StateOperator, model), adjoint = nothing)
    StateOperator(model; depends, solve = …, adjoint = nothing)

Forward state: a HyperFEM computational model built by the user (weak forms and solver are the user's) that is
solved again only when one of its dependencies changed.

- `deps` or `depends` (given once, mandatory): the objects the state depends on (one object or a tuple), typically
  `θ` for a forward state. Each implements [`state_version`](@ref); physical fields and other state operators are brought
  up to date first.
- `solve`: function `(model, ctx) -> result` that solves the model. The result is either the solution vector (taken
  as converged) or a tuple whose last element is the convergence flag: a `Bool`, or `1`/`0` as the `flagconv` of
  `(x, flagconv)` returned by HyperFEM's nonlinear models. Default for a `StaticLinearModel`:
  `solve!(m; Assembly = true)` ([`default_solve`](@ref)). Mandatory for any other model, for example
  `(m, ctx) -> solve!(m; stepping = …, RestartState = false)`; return the tuple, so that a load ramp that does not
  reach its final factor counts as not converged (an adjoint built on the tangent at the final factor would be
  inconsistent otherwise).

- `adjoint`: an [`AdjointProblem`](@ref), needed to differentiate the functionals that read the state: who solves the
  transposed system (an adjoint model with the transposed tangent, or [`SameMatrix`](@ref)`()`) and the forms
  `pᵀ ∂R/∂θ` of the state. Without it the state only gives values.

`get_state(S)` is the state field of the model, created once, so the forms of other models (the adjoint) can
capture it; it always holds the model's current solution vector.

Failures do not throw. When the solve does not converge, or a state this one depends on did not converge (then
this one is not solved), [`apply!`](@ref) returns `false`, [`is_converged`](@ref) is `false` and the functionals
that need the state return `NaN`. After a failure `get_state(S)` holds whatever the model left (for HyperFEM's
nonlinear models, the last converged load increment), not a solution. A failed solve is not repeated until a
dependency changes; the next solve starts from what the model holds, as decided by the user's `solve`
(`RestartState` in HyperFEM).
"""
mutable struct StateOperator{M,F,D,U,A} <: AbstractOperator
  model::M
  solve::F
  depends::D
  seen::Vector{Int}
  state::U
  version::Int
  converged::Bool
  solved::Bool
  adjoint::A
end

function StateOperator(model, deps = _Unset(); depends = _Unset(), solve = default_solve(StateOperator, model),
                       adjoint = nothing)
  d = _dependencies("StateOperator", deps, depends)
  d isa _Unset &&
    throw(ArgumentError("StateOperator: the dependencies are mandatory: StateOperator(model, θ) or depends = θ"))
  adjoint isa Union{Nothing,AdjointProblem} ||
    throw(ArgumentError("StateOperator: adjoint must be an AdjointProblem (AdjointProblem(model_adj; sensitivity) or AdjointProblem(SameMatrix(); sensitivity)); got $(typeof(adjoint))"))
  deps = _check_depends(_unique_deps(_as_depends(d)))
  StateOperator(model, solve, deps, fill(-1, length(deps)), HyperFEM.get_state(model), 0, false, false,
                _adjoint_cache(adjoint, model, deps))
end

has_adjoint(S::StateOperator) = S.adjoint !== nothing

state_version(S::StateOperator) = S.version
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

default_solve(::Type{<:StateOperator}, ::HyperFEM.StaticLinearModel) = _solve_state_linear
default_solve(::Type{<:LinearOperator}, ::HyperFEM.StaticLinearModel) = _solve_rhs_linear
# several right-hand sides: the state is not one field and the filter takes one vector
_several_rhs(model) =
  throw(ArgumentError("no default solve for a StaticLinearModel with several right-hand sides: pass your own solve"))
default_solve(::Type{<:StateOperator}, m::HyperFEM.StaticLinearModel{<:AbstractVector}) = _several_rhs(m)
default_solve(::Type{<:LinearOperator}, m::HyperFEM.StaticLinearModel{<:AbstractVector}) = _several_rhs(m)

_unique_deps(t::Tuple) = foldl((acc, d) -> any(x -> x === d, acc) ? acc : (acc..., d), t; init = ())

# Dirichlet values of the trial space of a model, or `nothing` when the model does not expose its spaces (only the
# abstract `get_spaces` of HyperFEM) or the space has no Dirichlet values to read
function dirichlet_values(model)
  _exposes_spaces(model) || return nothing
  _dirichlet_values(first(HyperFEM.get_spaces(model)))
end
const _AbstractGetSpaces = Tuple{typeof(HyperFEM.get_spaces),HyperFEM.ComputationalModels.ComputationalModel}
_exposes_spaces(model) =
  applicable(HyperFEM.get_spaces, model) && which(HyperFEM.get_spaces, Tuple{typeof(model)}).sig != _AbstractGetSpaces
_dirichlet_values(U) = applicable(get_dirichlet_dof_values, U) ? get_dirichlet_dof_values(U) : nothing
function _dirichlet_values(U::MultiFieldFESpace)
  vs = [_dirichlet_values(Ui) for Ui in U]
  any(v -> v === nothing, vs) ? nothing : reduce(vcat, vs; init = Float64[])
end

# the trial space of the model (if it exposes one) must have zero Dirichlet values
function _check_homogeneous_trial(model, who)
  v = dirichlet_values(model)
  v === nothing || all(iszero, v) ||
    throw(ArgumentError("$who: the trial space has non-zero Dirichlet values; build the adjoint on a homogeneous space"))
  nothing
end

Base.show(io::IO, S::StateOperator) =
  print(io, "StateOperator(", nameof(typeof(S.model)), ", ", length(S.depends), " dependencies, ",
        S.solved ? (S.converged ? "solved" : "not converged") : "not solved", ")")
