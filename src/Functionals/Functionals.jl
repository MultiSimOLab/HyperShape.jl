"""
    HyperShape.Functionals

Objectives and constraints. A [`Functional`](@ref) evaluates a quantity of interest and its derivative with respect
to the optimisation variables z, from forms written by the user with respect to the physical fields θ.
"""
module Functionals

using Gridap
using Gridap.FESpaces
using HyperFEM: HyperFEM, get_state, get_spaces
using ..Operators: StateOperator, apply!, state_version, dirichlet_values
import ..Operators: invalidate!
using ..Operators: get_model as HyperShape_get_model
using ..Variables: PhysicalFields, DesignFields, Identity, update_fields!, design_gradient!, get_designfields,
                   get_optvariables, is_linear_field

export Functional, Volume, functional_value, functional_gradient!

"""
    Functional(J, S::StateOperator; derivative, adjoint, scale, fields = nothing)
    Functional(J, θ::PhysicalFields; derivative, scale)

Objective or constraint.

With states (first form):
- `J(u, θ)`: value, a `DomainContribution` (`∫(…)dΩ`) or a number; `u = get_state(S)`.
- `derivative(u, p, θ)`: total derivative with respect to the physical fields, `∂J/∂θ − ∂R/∂θ` contracted with the
  adjoint `p`: a function `δ -> ∫(… * δ)dΩ` when θ has one field, or a named tuple of such functions
  (`(ρ = δ -> …, E = δ -> …)`), one per physical field it depends on.
- `S`: forward [`StateOperator`](@ref).
- `adjoint` (mandatory): the adjoint state, usually
  [`AdjointOperator`](@ref HyperShape.Operators.AdjointOperator)`(S, model)`, which solves `a(w, p) = ∂J/∂u[w]`
  for all `w` (with `a` the tangent of the forward problem) on a space with homogeneous Dirichlet values; it must
  depend on θ and on `S`. Or `:self` for a self-adjoint functional (then `p = u`, with the
  sign in `derivative`; valid only when the Dirichlet values of the state are zero, which is checked here and when
  the gradient is computed).
- `fields`: the physical fields θ. By default the only [`PhysicalFields`](@ref) among the dependencies of `S`;
  give it when `S` depends on none or on several.

Design only (second form, volume, perimeter): `J(θ)` and `derivative(θ)`.

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
  linear::Bool                  # design-only and affine in z (declared; checked when used as a constraint)
end

Base.show(io::IO, F::Functional) =
  print(io, "Functional(", F.S === nothing ? "design only" : "with a state", F.linear ? ", linear" : "",
        ", scale ", F.scale_mode === :initial ? (isnan(F.scale[]) ? ":initial (not set)" : "$(F.scale[]) (:initial)") :
        F.scale[], ")")

function _scale(scale)
  if scale === :initial
    return :initial, NaN
  elseif scale isa Real && isfinite(scale) && scale > 0
    return :fixed, Float64(scale)
  end
  throw(ArgumentError("Functional: scale must be a positive number, :initial or 1.0, got $scale"))
end

_depends_on(S::StateOperator, obj) = any(d -> d === obj, S.depends)

function _fields(S::StateOperator, fields)
  if fields === nothing
    θs = Tuple(d for d in S.depends if d isa PhysicalFields)   # the dependencies have no repetitions
    length(θs) == 1 || throw(ArgumentError(isempty(θs) ?
      "Functional: the forward state does not depend directly on any PhysicalFields; declare θ among its dependencies (StateOperator(model, θ))" :
      "Functional: the forward state depends on several PhysicalFields; pass fields = θ"))
    return θs[1]
  end
  fields isa PhysicalFields || throw(ArgumentError("Functional: fields must be a PhysicalFields"))
  _depends_on(S, fields) || throw(ArgumentError("Functional: the forward state must depend on θ (StateOperator(model, θ))"))
  fields
end

struct _NoAdjoint end

function Functional(J, S::StateOperator; derivative, adjoint = _NoAdjoint(), scale, fields = nothing)
  adjoint isa _NoAdjoint && throw(ArgumentError(
    "Functional: adjoint is mandatory: adjoint = AdjointOperator(S, model), or adjoint = :self for a self-adjoint functional"))
  mode, value = _scale(scale)
  θ = _fields(S, fields)
  A = if adjoint === :self
    _check_homogeneous(S, true)
    S
  elseif adjoint isa StateOperator
    adjoint === S && throw(ArgumentError("Functional: write adjoint = :self for a self-adjoint functional"))
    (_depends_on(adjoint, θ) && _depends_on(adjoint, S)) ||
      throw(ArgumentError("Functional: the adjoint state must depend on θ and on the forward state (use AdjointOperator(S, model))"))
    _check_homogeneous(adjoint, false)
    adjoint
  else
    throw(ArgumentError("Functional: adjoint must be a StateOperator (AdjointOperator(S, model)) or :self, got $(repr(adjoint))"))
  end
  Functional(J, derivative, S, A, θ, mode, Ref(value), [-1.0, -1.0, NaN], false)
end

function Functional(J, θ::PhysicalFields; derivative, scale, linear::Bool = false)
  mode, value = _scale(scale)
  Functional(J, derivative, nothing, nothing, θ, mode, Ref(value), [-1.0, -1.0, NaN], linear)
end

"""
    Volume(θ::PhysicalFields, dΩ; max_fraction, field = nothing, scale = Vmax)
    Volume(x::DesignFields, dΩ; max_fraction, field = nothing, scale = Vmax)

Volume constraint `g = ∫ θ.field dΩ - Vmax ≤ 0` with `Vmax = max_fraction ⋅ |Ω|` (`0 < max_fraction ≤ 1`, `|Ω|` the
measure of `dΩ`), a design-only [`Functional`](@ref) with its value and derivative written in. With physical fields θ, the volume of the physical density: nonlinear in z if its chain has a
[`TanhProjection`](@ref HyperShape.Variables.TanhProjection) projection, linear otherwise. With design fields x, the volume of
the design density (linear; the identity physical fields are built inside). `field` is mandatory when there are
several fields. `scale` divides the value and the derivative; default `Vmax`.
"""
function Volume(θ::PhysicalFields, dΩ; max_fraction::Real, field = nothing, scale = nothing)
  names = getfield(θ, :names)
  f = field === nothing ?
      (length(names) == 1 ? names[1] : throw(ArgumentError("Volume: several fields ($(join(names, ", "))); give field = …"))) :
      field
  f in names || throw(ArgumentError("Volume: no field named $f; available: $(join(names, ", "))"))
  0 < max_fraction <= 1 || throw(ArgumentError("Volume: max_fraction must be in (0, 1], got $max_fraction"))
  vmax = Float64(max_fraction) * sum(∫(1.0)dΩ)
  scale = scale === nothing ? vmax : scale
  J = θ -> sum(∫(getproperty(θ, f))dΩ) - vmax
  DJ = θ -> NamedTuple{(f,)}((δ -> ∫(δ)dΩ,))
  Functional(J, θ; derivative = DJ, scale, linear = is_linear_field(θ, f))
end

function Volume(x::DesignFields, dΩ; kwargs...)
  names = getfield(x, :names)
  θ = PhysicalFields(x; NamedTuple{names}(ntuple(_ -> Identity(), length(names)))...)
  Volume(θ, dΩ; kwargs...)
end

_total(v::Real) = Float64(v)
_total(v) = Float64(sum(v))

_state_version(F::Functional{FJ,FD,Nothing}) where {FJ,FD} = 0
_state_version(F::Functional) = state_version(F.S)

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
  vθ, vS = Float64(state_version(F.θ)), Float64(_state_version(F))
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
    throw(ArgumentError("Functional: derivative must return a named tuple of forms when θ has several fields ($(join(names, ", ")))"))
  NamedTuple{(names[1],)}((form,))
end

# the adjoint (or the state of a self-adjoint functional) must have homogeneous Dirichlet values
_check_homogeneous(F::Functional{FJ,FD,Nothing}) where {FJ,FD} = nothing
_check_homogeneous(F::Functional) = _check_homogeneous(F.A, F.A === F.S)
function _check_homogeneous(A::StateOperator, self::Bool)
  v = dirichlet_values(HyperShape_get_model(A))
  v === nothing || all(iszero, v) || throw(ArgumentError(self ?
    "Functional: adjoint = :self needs homogeneous Dirichlet values in the state; give an AdjointOperator on a homogeneous space" :
    "Functional: the trial space of the adjoint has non-zero Dirichlet values; build the adjoint on a homogeneous space"))
  nothing
end

"""
    functional_gradient!(gz, F::Functional, ctx) -> gz

Scaled derivative of the functional with respect to z, written into `gz`. Solves the forward state (if needed) and
the adjoint state, assembles `derivative` and pulls it back to z. Filled with `NaN` if a state did not converge.
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


# the next value is computed again (not taken from the cache)
invalidate!(F::Functional) = (F.cache[1] = -1.0; F)

end
