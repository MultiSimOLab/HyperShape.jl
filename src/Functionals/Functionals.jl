"""
    HyperShape.Functionals

Objectives and constraints. A [`Functional`](@ref) evaluates a quantity of interest and its derivative with respect
to the optimisation variables z, from forms written by the user with respect to the physical fields θ.
"""
module Functionals

using Gridap
using Gridap.FESpaces
using HyperFEM: HyperFEM, get_state, get_spaces
using ..Operators: StateOperator, apply!, version
using ..Operators: get_model as HyperShape_get_model
using ..Variables: PhysicalFields, update_fields!, design_gradient!, get_designfields, get_optvariables

export Functional, functional_value, functional_gradient!

"""
    Functional(Jform, DJform, S, A, θ; scale)
    Functional(Jform, DJform, θ; scale)

Objective or constraint.

With states (first form):
- `Jform(u, θ)`: value, a `DomainContribution` (`∫(…)dΩ`) or a number; `u = get_state(S)`.
- `DJform(u, p, θ)`: total derivative with respect to the physical fields, `∂J/∂θ − ∂R/∂θ` contracted with the
  adjoint `p = get_state(A)`: a function `δ -> ∫(… * δ)dΩ` when θ has one field, or a named tuple of such functions
  (`(ρ = δ -> …, E = δ -> …)`), one per physical field it depends on.
- `S`: forward [`StateOperator`](@ref); `A`: adjoint state operator, which solves `a(w, p) = ∂J/∂u[w]` for all `w`
  (with `a` the tangent of the forward problem) on a space with homogeneous Dirichlet values; `S` must depend on θ
  and `A` on θ and `S`. For a self-adjoint functional pass `A = S` (then `p = u`; valid only when the Dirichlet
  values of the state are zero) and include the sign in `DJform`. Non-zero Dirichlet values in the trial space of
  the adjoint are detected when the gradient is computed.

Design only (second form, volume, perimeter): `Jform(θ)` and `DJform(θ)`.

`scale` is mandatory: a positive number, `:initial` (the absolute value of the first finite evaluation; an error if
it is zero) or `1.0` for no scaling. The value and the derivative are divided by it.

The value is kept until θ or a state changes; the gradient is assembled at every call (one adjoint solve only when
the design changed, one transposed filter solve per filter and one assembly per call).

If a state does not converge, [`functional_value`](@ref) returns `NaN` and [`functional_gradient!`](@ref) fills the
gradient with `NaN` (the optimisers treat it as a failed trial).
"""
struct Functional{FJ,FD,S,A,T<:PhysicalFields}
  J::FJ
  DJ::FD
  S::S
  A::A
  θ::T
  scale_mode::Symbol            # :fixed or :initial
  scale::Base.RefValue{Float64} # NaN while :initial is not set
  cache::Vector{Float64}        # [version of θ, version of S, raw value] of the last evaluation
end

function _scale(scale)
  if scale === :initial
    return :initial, NaN
  elseif scale isa Real && isfinite(scale) && scale > 0
    return :fixed, Float64(scale)
  end
  throw(ArgumentError("Functional: scale must be a positive number, :initial or 1.0, got $scale"))
end

_depends_on(S::StateOperator, obj) = any(d -> d === obj, S.depends)

function Functional(Jform, DJform, S::StateOperator, A::StateOperator, θ::PhysicalFields; scale)
  mode, value = _scale(scale)
  _depends_on(S, θ) || throw(ArgumentError("Functional: the forward state must depend on θ (depends = θ)"))
  if A !== S
    (_depends_on(A, θ) && _depends_on(A, S)) ||
      throw(ArgumentError("Functional: the adjoint state must depend on θ and on the forward state (depends = (θ, S))"))
  end
  Functional(Jform, DJform, S, A, θ, mode, Ref(value), [-1.0, -1.0, NaN])
end

function Functional(Jform, DJform, θ::PhysicalFields; scale)
  mode, value = _scale(scale)
  Functional(Jform, DJform, nothing, nothing, θ, mode, Ref(value), [-1.0, -1.0, NaN])
end

_total(v::Real) = Float64(v)
_total(v) = Float64(sum(v))

_state_version(F::Functional{FJ,FD,Nothing}) where {FJ,FD} = 0
_state_version(F::Functional) = version(F.S)

_compute_value(F::Functional{FJ,FD,Nothing}, ctx) where {FJ,FD} = _total(F.J(F.θ))
function _compute_value(F::Functional, ctx)
  apply!(F.S, ctx) || return NaN
  _total(F.J(get_state(F.S), F.θ))
end

# value without scaling, kept until θ or the state changes
function _raw_value(F::Functional, ctx)::Float64
  update_fields!(F.θ, ctx)
  F.S === nothing || apply!(F.S, ctx)
  c = F.cache
  vθ, vS = Float64(version(F.θ)), Float64(_state_version(F))
  (c[1] == vθ && c[2] == vS) && return c[3]
  v = _compute_value(F, ctx)
  c[1], c[2], c[3] = vθ, vS, v
  v
end

function _set_scale!(F::Functional, v)
  if F.scale_mode === :initial && isnan(F.scale[])
    isfinite(v) || return F
    v == 0 && throw(ArgumentError("Functional: scale = :initial with a zero first value; give the scale explicitly"))
    F.scale[] = abs(v)
  end
  F
end

"""
    functional_value(F::Functional, ctx) -> Float64

Scaled value of the functional at the current design (states solved again only if the design changed). `NaN` if a
state did not converge.
"""
function functional_value(F::Functional, ctx)::Float64
  v = _raw_value(F, ctx)
  _set_scale!(F, v)
  v / F.scale[]
end

_forms(F::Functional{FJ,FD,Nothing}, ctx) where {FJ,FD} = F.DJ(F.θ)
function _forms(F::Functional, ctx)
  u = get_state(F.S)
  p = F.A === F.S ? u : get_state(F.A)
  F.DJ(u, p, F.θ)
end

function _as_named(forms::NamedTuple, θ)
  forms
end
function _as_named(form, θ)
  names = propertynames(θ)
  length(names) == 1 ||
    throw(ArgumentError("Functional: DJform must return a named tuple of forms when θ has several fields ($(join(names, ", ")))"))
  NamedTuple{(names[1],)}((form,))
end

# the adjoint (or the state of a self-adjoint functional) must have homogeneous Dirichlet values
_check_homogeneous(F::Functional{FJ,FD,Nothing}) where {FJ,FD} = nothing
function _check_homogeneous(F::Functional)
  m = HyperShape_get_model(F.A)
  applicable(get_spaces, m) || return nothing
  U = first(get_spaces(m))
  applicable(get_dirichlet_dof_values, U) || return nothing
  all(iszero, get_dirichlet_dof_values(U)) || throw(ArgumentError(F.A === F.S ?
    "Functional: A = S needs homogeneous Dirichlet values in the state; give an adjoint state on a homogeneous space" :
    "Functional: the trial space of the adjoint has non-zero Dirichlet values; build the adjoint on a homogeneous space"))
  nothing
end

"""
    functional_gradient!(gz, F::Functional, ctx) -> gz

Scaled derivative of the functional with respect to z, written into `gz`. Solves the forward state (if needed) and
the adjoint state, assembles `DJform` and pulls it back to z. Filled with `NaN` if a state did not converge.
"""
function functional_gradient!(gz, F::Functional, ctx)
  update_fields!(F.θ, ctx)
  if F.S !== nothing
    if !apply!(F.S, ctx) || (F.A !== F.S && !apply!(F.A, ctx))
      fill!(gz, NaN)
      return gz
    end
  end
  _check_homogeneous(F)
  isnan(F.scale[]) && _set_scale!(F, _raw_value(F, ctx))
  design_gradient!(gz, F.θ, _as_named(_forms(F, ctx), F.θ), ctx; key = objectid(F.scale))
  gz ./= F.scale[]
  gz
end

end
