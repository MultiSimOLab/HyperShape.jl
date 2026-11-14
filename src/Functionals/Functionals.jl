"""
    HyperShape.Functionals

Objectives and constraints. A [`Functional`](@ref) evaluates a quantity of interest and its derivative with respect
to the optimisation variables z, from forms written by the user with respect to the physical fields θ.
"""
module Functionals

using Gridap
using HyperFEM: HyperFEM, get_state
using ..Operators: StateOperator, apply!
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
- `S`: forward [`StateOperator`](@ref); `A`: adjoint state operator. For a self-adjoint functional pass `A = S`
  (then `p = u`) and include the sign in `DJform`.

Design only (second form, volume, perimeter): `Jform(θ)` and `DJform(θ)`.

`scale` is mandatory: a positive number, `:initial` (the absolute value of the first evaluation) or `1.0` for no
scaling. The value and the derivative are divided by it.

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
end

function _scale(scale)
  if scale === :initial
    return :initial, NaN
  elseif scale isa Real && isfinite(scale) && scale > 0
    return :fixed, Float64(scale)
  end
  throw(ArgumentError("Functional: scale must be a positive number, :initial or 1.0, got $scale"))
end

function Functional(Jform, DJform, S::StateOperator, A::StateOperator, θ::PhysicalFields; scale)
  mode, value = _scale(scale)
  Functional(Jform, DJform, S, A, θ, mode, Ref(value))
end

function Functional(Jform, DJform, θ::PhysicalFields; scale)
  mode, value = _scale(scale)
  Functional(Jform, DJform, nothing, nothing, θ, mode, Ref(value))
end

_total(v::Real) = Float64(v)
_total(v) = Float64(sum(v))

_raw_value(F::Functional{FJ,FD,Nothing}, ctx) where {FJ,FD} = _total(F.J(F.θ))
function _raw_value(F::Functional, ctx)
  apply!(F.S, ctx) || return NaN
  _total(F.J(get_state(F.S), F.θ))
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
function functional_value(F::Functional, ctx)
  update_fields!(F.θ, ctx)
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
  isnan(F.scale[]) && _set_scale!(F, _raw_value(F, ctx))
  design_gradient!(gz, F.θ, _as_named(_forms(F, ctx), F.θ), ctx)
  gz ./= F.scale[]
  gz
end

end
