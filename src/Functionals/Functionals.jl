"""
    HyperShape.Functionals

Objectives and constraints. A [`Functional`](@ref) evaluates a quantity of interest and its derivative with respect
to the optimisation variables z, from forms written by the user with respect to the physical fields θ.
"""
module Functionals

using Gridap
using Gridap.FESpaces
using HyperFEM: HyperFEM, get_state, get_spaces
using ..Operators: StateOperator, apply!, state_version, dirichlet_values, SelfAdjoint, has_adjoint, reset_adjoint!,
                   is_seeded, add_seed!, add_self_seed!, solve_adjoint!, adjoint_sensitivity
import ..Operators: invalidate!
using ..Operators: get_model as HyperShape_get_model
using ..Variables: PhysicalFields, DesignFields, Identity, update_fields!, design_gradient!, get_designfields,
                   get_optvariables, is_linear_field, reset_sink!, add_forms!, finish_sink!, DesignMeasures

export AbstractFunctional, Functional, Constraint, Volume, LinearInState, functional_value, functional_gradient!

"""
    LinearInState()

`dJdu` of a [`Functional`](@ref) whose value `J(u, θ)` is linear in the state `u` (no constant term), such as
`J = (u, θ) -> ∫(g ⋅ u)dΓ`: its derivative in the direction `w` is `J(w, θ)`, so it is not written twice.
`J` must return a `DomainContribution`; `check_consistency` detects a functional that is not linear.
"""
struct LinearInState end

# u -> (w -> J(w, θ))
struct _LinearDerivative{F,T}
  J::F
  θ::T
end
(d::_LinearDerivative)(u) = w -> d.J(w, d.θ)

"""
    AbstractFunctional

Supertype of the objectives and constraints. What the optimisation oracle uses of a subtype `F`:
[`functional_value`](@ref), [`functional_gradient!`](@ref) (alone and in a weighted tuple), `get_scale(F)`,
`is_design_only(F)` (no state is solved for it), `is_linear(F)` (declared affine in z), `physical_fields(F)`,
`responses(F)` (the states it reads) and `invalidate!(F)`. The internals of the reverse sweep (`_seed!`,
`_pull_back!`) will change with chains of states.
"""
abstract type AbstractFunctional end

"""
    Functional(J, S::StateOperator; dJdu, dJdθ = nothing, scale, fields = nothing, parameters = ())
    Functional(J, θ::PhysicalFields; dJdθ, scale, linear = false, parameters = ())

Objective or constraint.

With a state (first form), `S` a [`StateOperator`](@ref) built with an
[`AdjointProblem`](@ref HyperShape.Operators.AdjointProblem) (it declares how to solve the transposed system and the forms
`pᵀ ∂R/∂θ` of the state):
- `J(u, θ)`: value, a `DomainContribution` (`∫(…)dΩ`) or a number; `u = get_state(S)`.
- `dJdu` (mandatory): the partial derivative with respect to the state, `u -> (w -> ∫(…)dΩ)` (a linear form in the
  test function `w` of the adjoint), [`LinearInState`](@ref)`()` when `J` is linear in `u` (a mean displacement, a
  temperature at a probe: then `dJdu` is `J` itself with `w` in place of `u`), or
  [`SelfAdjoint`](@ref HyperShape.Operators.SelfAdjoint)`()` when it is the load of a linear and symmetric state (the
  compliance): then `p = u` and no adjoint is solved.
- `dJdθ`: the partial derivative with respect to the physical fields, `(u, θ) -> (δ -> ∫(… * δ)dΩ)`, or a named tuple
  of such forms (`(ρ = δ -> …, E = δ -> …)`) when θ has several fields; `nothing` (default) when `J` does not read θ
  directly.
- `fields`: the physical fields θ, the only [`PhysicalFields`](@ref) among the dependencies of `S` (deduced; a state
  that depends on several is not supported yet).

The derivative with respect to θ is `dJdθ − pᵀ ∂R/∂θ`, with `p` the adjoint of `S` seeded by `dJdu`: the signs are
put by HyperShape. A state that depends on another state is not supported yet (an error here): the sensitivity
through the upstream state would be missing.

Design only (second form, volume, perimeter): `J(θ)` and `dJdθ(θ)`, the derivative with respect to θ.

`scale` is mandatory: a positive number, `:initial` (the absolute value of the first finite evaluation; an error if
it is zero), `:bound` (for a constraint: the largest finite bound of its [`Constraint`](@ref) in absolute value) or
`1.0` for no scaling. The value and the derivative are divided by it.

`parameters`: [`Parameter`](@ref HyperShape.Operators.Parameter)s (or other objects with a `state_version`) that `J`
reads directly, not through θ or a state (a weight that is continued, for example): the value is computed again
when they change. A parameter read by a model goes in the dependencies of its operator instead; one read by a map,
in the `parameters` of the `PointwiseMap`. `linear` (design-only form) declares a functional affine in z, checked
when it is used as a constraint.

The value is kept until θ, a state or a parameter changes. Each gradient is one reverse sweep: the seeds of the
functional, one adjoint solve of the state (a back substitution after the first gradient of a design; none for
`SelfAdjoint()`), one transposed solve per filter and the assemblies of the forms.

If a state does not converge, [`functional_value`](@ref) returns `NaN` and [`functional_gradient!`](@ref) fills the
gradient with `NaN` (the optimisers treat it as a failed trial).
"""
struct Functional{FJ,FD,FU,S,T<:PhysicalFields,P<:Tuple} <: AbstractFunctional
  J::FJ
  DJ::FD                        # design only: dJdθ(θ); with a state: dJdθ(u, θ) or nothing
  dJdu::FU                      # with a state: u -> (w -> ∫…) or SelfAdjoint(); nothing for design only
  S::S
  θ::T
  scale_mode::Symbol            # :fixed, :initial or :bound (set by Constraint)
  scale::Base.RefValue{Float64} # NaN while :initial is not set
  cache::Vector{Float64}        # [version of θ, version of S, version of the parameters, raw value] of the last evaluation
  linear::Bool                  # design-only and affine in z (declared; checked when used as a constraint)
  parameters::P
end

function _parameters(ps)
  t = ps isa Union{Tuple,AbstractVector} ? Tuple(ps) : (ps,)
  for q in t
    applicable(state_version, q) ||
      throw(ArgumentError("Functional: a parameter must implement state_version (a Parameter); got $(typeof(q))"))
  end
  t
end

Base.show(io::IO, F::Functional) =
  print(io, "Functional(", F.S === nothing ? "design only" : "with a state", F.linear ? ", linear" : "",
        ", scale ", F.scale_mode === :initial ? (isnan(F.scale[]) ? ":initial (not set)" : "$(F.scale[]) (:initial)") :
        F.scale[], ")")

function _scale(scale)
  if scale === :initial || scale === :bound
    return scale, NaN
  elseif scale isa Real && isfinite(scale) && scale > 0
    return :fixed, Float64(scale)
  end
  throw(ArgumentError("Functional: scale must be a positive number, :initial, :bound (in a Constraint) or 1.0, got $scale"))
end

get_scale(F::Functional) = F.scale[]
is_design_only(F::Functional) = F.S === nothing
is_linear(F::Functional) = F.linear
physical_fields(F::Functional) = F.θ
responses(F::Functional) = F.S === nothing ? () : (F.S,)

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

struct _Unused end

function Functional(J, S::StateOperator; dJdu = _Unused(), dJdθ = nothing, scale, fields = nothing, parameters = (),
                    derivative = _Unused(), adjoint = _Unused())
  (derivative isa _Unused && adjoint isa _Unused) || throw(ArgumentError(
    "Functional: `derivative` and `adjoint` are gone for functionals with a state: declare the adjoint in the state " *
    "(StateOperator(model, θ; adjoint = AdjointProblem(model_adj or SymmetricState(); sensitivity = p -> pᵀ∂R/∂θ))) and give " *
    "here dJdu = u -> (w -> ∫(∂J/∂u ⋅ w)dΩ) (or SelfAdjoint()) and, if J reads θ, dJdθ = (u, θ) -> (δ -> ∫(…)dΩ)"))
  dJdu isa _Unused && throw(ArgumentError(
    "Functional: dJdu is mandatory: dJdu = u -> (w -> ∫(…)dΩ), or SelfAdjoint() for the compliance of a linear and symmetric state"))
  has_adjoint(S) || throw(ArgumentError(
    "Functional: the state has no adjoint: StateOperator(model, θ; adjoint = AdjointProblem(model_adj; sensitivity)) " *
    "or AdjointProblem(SymmetricState(); sensitivity) for a linear and symmetric state"))
  any(d -> d isa StateOperator, S.depends) && throw(ArgumentError(
    "Functional: the state depends on another state; the sensitivity through it is not supported yet " *
    "(chains of states come in a later version)"))
  dJdu isa SelfAdjoint || S.adjoint.op !== nothing || throw(ArgumentError(
    "Functional: the AdjointProblem of the state has no operator (only SelfAdjoint functionals); give it an adjoint model or SymmetricState()"))
  count(d -> d isa PhysicalFields, S.depends) <= 1 || throw(ArgumentError(
    "Functional: the state depends on several PhysicalFields; the sensitivity through all of them is not supported yet"))
  mode, value = _scale(scale)
  θ = _fields(S, fields)
  dJdu isa LinearInState && (dJdu = _LinearDerivative(J, θ))
  if dJdu isa SelfAdjoint
    HyperShape_get_model(S) isa HyperFEM.StaticLinearModel || throw(ArgumentError(
      "Functional: dJdu = SelfAdjoint() needs a linear state (a StaticLinearModel): p = u is not the adjoint of a nonlinear problem"))
    _check_homogeneous(S)
  end
  Functional(J, dJdθ, dJdu, S, θ, mode, Ref(value), [-1.0, -1.0, -1.0, NaN], false, _parameters(parameters))
end

function Functional(J, θ::PhysicalFields; dJdθ = _Unused(), scale, linear::Bool = false, parameters = (),
                    derivative = _Unused())
  derivative isa _Unused || throw(ArgumentError("Functional: `derivative` is now `dJdθ = θ -> (δ -> ∫(…)dΩ)`"))
  dJdθ isa _Unused && throw(ArgumentError("Functional: dJdθ is mandatory for a design-only functional: dJdθ = θ -> (δ -> ∫(…)dΩ)"))
  mode, value = _scale(scale)
  Functional(J, dJdθ, nothing, nothing, θ, mode, Ref(value), [-1.0, -1.0, -1.0, NaN], linear, _parameters(parameters))
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
  Functional(J, θ; dJdθ = DJ, scale, linear = is_linear_field(θ, f))
end

Volume(θ::PhysicalFields, dΩ::DesignMeasures; kwargs...) = Volume(θ, dΩ.domain; kwargs...)

function Volume(x::DesignFields, dΩ; kwargs...)
  names = getfield(x, :names)
  θ = PhysicalFields(x; NamedTuple{names}(ntuple(_ -> Identity(), length(names)))...)
  Volume(θ, dΩ; kwargs...)
end

_total(v::Real) = Float64(v)
_total(v) = Float64(sum(v))

_state_version(F::Functional{FJ,FD,FU,Nothing}) where {FJ,FD,FU} = 0
_parameter_version(F::Functional) = sum(state_version, F.parameters; init = 0)
_state_version(F::Functional) = state_version(F.S)

_compute_value(F::Functional{FJ,FD,FU,Nothing}, ctx) where {FJ,FD,FU} = _total(F.J(F.θ))
function _compute_value(F::Functional, ctx)
  apply!(F.S, ctx) || return NaN
  _total(F.J(get_state(F.S), F.θ))
end

# value without scaling, kept until θ or the state changes
function _raw_value(F::Functional, ctx)::Float64
  update_fields!(F.θ, ctx)
  F.S === nothing || apply!(F.S, ctx)
  c = F.cache
  vθ, vS, vP = Float64(state_version(F.θ)), Float64(_state_version(F)), Float64(_parameter_version(F))
  (c[1] == vθ && c[2] == vS && c[3] == vP) && return c[4]
  v = _compute_value(F, ctx)
  c[1], c[2], c[3], c[4] = vθ, vS, vP, v
  v
end

function _set_scale!(F::Functional, v)
  F.scale_mode === :bound && isnan(F.scale[]) &&
    throw(ArgumentError("Functional: scale = :bound takes the scale from the bounds of a Constraint; use it inside Constraint(F; max, min, equal)"))
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

function _as_named(forms::NamedTuple, θ)
  forms
end
function _as_named(form, θ)
  names = propertynames(θ)
  length(names) == 1 ||
    throw(ArgumentError("Functional: the derivative with respect to θ must be a named tuple of forms when θ has several fields ($(join(names, ", ")))"))
  NamedTuple{(names[1],)}((form,))
end

# a self-adjoint functional needs homogeneous Dirichlet values in the state
function _check_homogeneous(S::StateOperator)
  v = dirichlet_values(HyperShape_get_model(S))
  v === nothing || all(iszero, v) || throw(ArgumentError(
    "Functional: dJdu = SelfAdjoint() needs homogeneous Dirichlet values in the state; give dJdu as a form and an AdjointProblem model on a homogeneous space"))
  nothing
end

# ---------------------------------------------------------------------------------------------------------------
# Reverse sweep: seed (each functional adds ∂J/∂u to the adjoint of its state and ∂J/∂θ to the design sink), pull
# back (each seeded state solves its adjoint once and adds −pᵀ∂R/∂θ), finish (transposed filters once, x → z).

_key(F::Functional) = objectid(F.scale)

_seed!(θ, F::Functional{FJ,FD,FU,Nothing}, w, ctx) where {FJ,FD,FU} =
  iszero(w) ? nothing : (add_forms!(θ, _as_named(F.DJ(θ), θ), w, ctx; key = _key(F)); nothing)

function _seed!(θ, F::Functional, w, ctx)
  S = F.S
  u = get_state(S)
  if F.dJdu isa SelfAdjoint
    _check_homogeneous(S)
    add_self_seed!(S, w)
  else
    add_seed!(S, F.dJdu(u), w, (_key(F), :dJdu))
  end
  F.DJ === nothing || iszero(w) || add_forms!(θ, _as_named(F.DJ(u, θ), θ), w, ctx; key = _key(F))
  nothing
end

_reset_state!(F::Functional{FJ,FD,FU,Nothing}) where {FJ,FD,FU} = nothing
_reset_state!(F::Functional) = (reset_adjoint!(F.S); nothing)

_pull_back!(θ, F::Functional{FJ,FD,FU,Nothing}, ctx) where {FJ,FD,FU} = nothing
function _pull_back!(θ, F::Functional, ctx)
  S = F.S
  is_seeded(S) || return nothing            # already pulled back in this sweep (several functionals on one state)
  p = solve_adjoint!(S, ctx)
  add_forms!(θ, _as_named(adjoint_sensitivity(S, p), θ), -1.0, ctx; key = objectid(S.adjoint))
  nothing
end

_converged!(F::Functional{FJ,FD,FU,Nothing}, ctx) where {FJ,FD,FU} = true
_converged!(F::Functional, ctx) = apply!(F.S, ctx)

"""
    functional_gradient!(gz, F::Functional, ctx) -> gz

Scaled derivative of the functional with respect to z, written into `gz`: one reverse sweep (solves the forward
state if needed, the adjoint of the state, the transposed filters). Filled with `NaN` if a state did not converge.
"""
function functional_gradient!(gz, F::Functional, ctx)
  θ = F.θ
  update_fields!(θ, ctx)
  _converged!(F, ctx) || return fill!(gz, NaN)
  isnan(F.scale[]) && _set_scale!(F, _raw_value(F, ctx))
  reset_sink!(θ)
  _reset_state!(F)
  _seed!(θ, F, 1.0, ctx)
  _pull_back!(θ, F, ctx)
  finish_sink!(gz, θ, ctx)
  gz ./= F.scale[]
  gz
end

"""
    functional_gradient!(gz, Fs::Tuple, w, ctx) -> gz

Derivative with respect to z of `Σᵢ wᵢ Fᵢ` (each `Fᵢ` scaled) in one reverse sweep per set of physical fields θ:
every state is solved once whatever the number of functionals that read it (one adjoint solve per state, not per
functional) and every filter is solved once. The seeds are `wᵢ sᵣ/sᵢ` and the result is divided by `sᵣ`, the scale
of the first functional of the sweep, so that a tuple of one functional gives the same bits as
`functional_gradient!(gz, F, ctx)`. Functionals on different θ (a `Volume` of the design fields) are summed sweep by
sweep. A functional with weight zero costs nothing (its state is not solved). Filled with `NaN` if a state did not
converge.
"""
function functional_gradient!(gz, Fs::Tuple{Vararg{AbstractFunctional}}, w, ctx)
  isempty(Fs) && throw(ArgumentError("functional_gradient!: no functionals"))
  length(w) == length(Fs) || throw(DimensionMismatch("functional_gradient!: $(length(w)) weights for $(length(Fs)) functionals"))
  θ = first(Fs).θ
  _all_share(Fs, θ) && return _sweep!(gz, θ, Fs, w, ctx)
  # several θ: one sweep per θ, added
  fill!(gz, 0.0)
  work = similar(gz)
  θs = unique(F -> objectid(F.θ), collect(Any, Fs))
  for G in θs
    idx = [i for i in eachindex(Fs) if Fs[i].θ === G.θ]
    all(i -> iszero(w[i]), idx) && continue            # a group with weight zero costs nothing
    _sweep!(work, G.θ, Tuple(Fs[idx]), [w[i] for i in idx], ctx)
    gz .+= work
  end
  gz
end

function _sweep!(gz, θ, Fs, w, ctx)
  update_fields!(θ, ctx)
  _all_converged!(Fs, w, 1, ctx) || return fill!(gz, NaN)
  _set_scales!(Fs, w, 1, ctx)
  s1 = first(Fs).scale[]
  sref = isfinite(s1) ? s1 : 1.0          # the first one has weight zero and no scale yet: any reference will do
  reset_sink!(θ)
  _reset_states!(Fs)
  _seed_all!(θ, Fs, w, sref, 1, ctx)
  _pull_back_all!(θ, Fs, ctx)
  finish_sink!(gz, θ, ctx)
  gz ./= sref
  gz
end

# loops over the (heterogeneous) tuple of functionals, unrolled
_all_share(::Tuple{}, θ) = true
_all_share(t::Tuple, θ) = first(t).θ === θ && _all_share(Base.tail(t), θ)
# a functional with weight zero is skipped (its state is not solved, it is not seeded): the tuple keeps one type
# whatever the weights, so a change of the active set of an augmented Lagrangian compiles nothing
_all_converged!(::Tuple{}, w, i, ctx) = true
_all_converged!(t::Tuple, w, i, ctx) =
  (iszero(w[i]) || _converged!(first(t), ctx)) & _all_converged!(Base.tail(t), w, i + 1, ctx)
_set_scales!(::Tuple{}, w, i, ctx) = nothing
_set_scales!(t::Tuple, w, i, ctx) = (F = first(t); (!iszero(w[i]) && isnan(F.scale[])) && _set_scale!(F, _raw_value(F, ctx));
                                     _set_scales!(Base.tail(t), w, i + 1, ctx))
_reset_states!(::Tuple{}) = nothing
_reset_states!(t::Tuple) = (_reset_state!(first(t)); _reset_states!(Base.tail(t)))
_seed_all!(θ, ::Tuple{}, w, sref, i, ctx) = nothing
_seed_all!(θ, t::Tuple, w, sref, i, ctx) =
  (F = first(t); iszero(w[i]) || _seed!(θ, F, Float64(w[i]) * sref / F.scale[], ctx);
   _seed_all!(θ, Base.tail(t), w, sref, i + 1, ctx))
_pull_back_all!(θ, ::Tuple{}, ctx) = nothing
_pull_back_all!(θ, t::Tuple, ctx) = (_pull_back!(θ, first(t), ctx); _pull_back_all!(θ, Base.tail(t), ctx))

# the next value is computed again (not taken from the cache)
invalidate!(F::Functional) = (F.cache[1] = -1.0; F)

"""
    Constraint(F::AbstractFunctional; max = nothing, min = nothing, equal = nothing)

A constraint on the functional `F`, in the units of `F` (before scaling): `F ≤ max`, `F ≥ min`, `min ≤ F ≤ max` or
`F = equal`. Without bounds, `F ≤ 0`; a functional given alone in the constraints of an [`Optimization`](@ref HyperShape.Methods.Optimization)
is `Constraint(F; max = 0.0)` (as `Volume`). With `scale = :bound` in `F`, its scale is the largest finite bound in
absolute value, so the bound is written once:

```julia
Tip = Functional((u, θ) -> ∫(g ⋅ u)dΓo, S; dJdu = u -> (w -> ∫(g ⋅ w)dΓo), scale = :bound)
problem = Optimization(objective = C, constraints = (volume = Vol, tip = Constraint(Tip; max = 0.25)))
```

The optimiser sees `g = (F − r)/s`, with `s` the scale of `F` and `r` the upper bound (the lower one if there is no
upper), and its bounds `[(min − r)/s, (max − r)/s]`. The scale belongs to the functional: with `scale = :bound`, build
one functional per `Constraint` (a second `Constraint` with another bound on the same functional throws).
"""
struct Constraint{F<:AbstractFunctional}
  F::F
  lo::Float64
  hi::Float64
end

function Constraint(F::AbstractFunctional; max = nothing, min = nothing, equal = nothing)
  if equal !== nothing
    (max === nothing && min === nothing) || throw(ArgumentError("Constraint: equal excludes max and min"))
    lo = hi = Float64(equal)
  else
    (max === nothing && min === nothing) && (max = 0.0)        # a functional alone: F ≤ 0
    lo = min === nothing ? -Inf : Float64(min)
    hi = max === nothing ? Inf : Float64(max)
  end
  (isnan(lo) || isnan(hi) || lo > hi || isinf(lo) && lo > 0 || isinf(hi) && hi < 0 || !(isfinite(lo) || isfinite(hi))) &&
    throw(ArgumentError("Constraint: need min ≤ max with at least one finite bound, got min = $lo, max = $hi"))
  if F isa Functional && F.scale_mode === :bound
    b = isfinite(lo) && isfinite(hi) ? Base.max(abs(lo), abs(hi)) : isfinite(hi) ? abs(hi) : abs(lo)
    b > 0 || throw(ArgumentError("Constraint: scale = :bound with zero bounds; give the scale explicitly"))
    # the scale belongs to the functional: a second Constraint on it must give the same one
    (isnan(F.scale[]) || F.scale[] == b) ||
      throw(ArgumentError("Constraint: this functional (scale = :bound) already has the scale $(F.scale[]) from " *
                          "another Constraint; build one functional per Constraint"))
    F.scale[] = b
  end
  Constraint(F, lo, hi)
end

Constraint(c::Constraint) = c

# reference value r subtracted from F (units of F): the upper bound, or the lower one
_reference(c::Constraint) = isfinite(c.hi) ? c.hi : c.lo

Base.show(io::IO, c::Constraint) =
  print(io, "Constraint(", c.lo == c.hi ? "= $(c.hi)" : c.lo == -Inf ? "≤ $(c.hi)" : c.hi == Inf ? "≥ $(c.lo)" :
        "in [$(c.lo), $(c.hi)]", ")")

end
