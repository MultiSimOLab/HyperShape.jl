# Maps between levels. A map is a specification; applied to an input field it creates a stage that owns its output
# (a fixed object updated in place, so weak forms can capture it) and knows how to pull a sensitivity back.
#
# Stage interface:
#   output(s)                 the output field (FEFunction or CellField)
#   update_stage!(s, ctx)     recompute the output from the input (in place)
#   pullback(s, L, ctx)       L: δ_out ↦ DomainContribution, linear form on the output; returns the linear form on the input
#   parameter_version(s)      counter of the parameters of the map (β of a Heaviside); 0 if it has none

"""
    FieldMap

Supertype of the maps x → θ. Compose them with `∘` in the mathematical order: `(g ∘ f)(x) = g(f(x))`.
"""
abstract type FieldMap end

"""
    Identity()

Identity map (z → x or x → θ).
"""
struct Identity <: FieldMap end

struct ComposedMap{T<:Tuple} <: FieldMap
  maps::T   # in the order of application
end

_maps(m::ComposedMap) = m.maps
_maps(m::FieldMap) = (m,)
Base.:∘(g::FieldMap, f::FieldMap) = ComposedMap((_maps(f)..., _maps(g)...))

struct IdentityStage{I}
  input::I
end
_stage(::Identity, input) = IdentityStage(input)
output(s::IdentityStage) = s.input
update_stage!(s::IdentityStage, ctx) = s
pullback(s::IdentityStage, L, ctx) = L
parameter_version(s) = 0

"""
    HelmholtzFilter(op::LinearOperator; measure)

PDE filter (Lazarov and Sigmund, 2011): the filtered field ρ̃ solves `a(ρ̃, v) = ∫(x v) dΩ` for all `v`, where
`a` is the bilinear form of the model of `op` (for example `a(ρ̃, v) = ∫(r^2 ∇(ρ̃)⋅∇(v) + ρ̃ v)dΩ`, solver chosen by
the user) and `dΩ = measure`. The output is a field of the trial space of that model (usually P1), also for a P0
design field. The bilinear form must be symmetric: the same solve gives the transpose for the sensitivities.

Cost: one assembly of a load vector and one solve to filter, and the same to pull a sensitivity back.
"""
struct HelmholtzFilter{O<:LinearOperator,M} <: FieldMap
  op::O
  measure::M
end
HelmholtzFilter(op::LinearOperator; measure) = HelmholtzFilter(op, measure)

struct FilterStage{O,I,M,F,W,V,A}
  op::O
  input::I
  measure::M
  out::F       # filtered field
  w::W         # adjoint field (pullback)
  b::Vector{Float64}
  V::V         # test space of the filter
  assem::A
end

function _stage(f::HelmholtzFilter, input)
  U, V = get_spaces(f.op.model)
  out = FEFunction(U, zeros(num_free_dofs(U)))
  w = FEFunction(U, zeros(num_free_dofs(U)))
  b = zeros(num_free_dofs(V))
  FilterStage(f.op, input, f.measure, out, w, b, V, SparseMatrixAssembler(U, V))
end

output(s::FilterStage) = s.out

function update_stage!(s::FilterStage, ctx)
  dΩ, xin = s.measure, s.input
  fill!(s.b, 0)
  assemble_vector!(v -> ∫(xin * v)dΩ, s.b, s.assem, s.V)
  solve_linear!(get_free_dof_values(s.out), s.op, s.b, ctx)
  s
end

function pullback(s::FilterStage, L, ctx)
  fill!(s.b, 0)
  assemble_vector!(L, s.b, s.assem, s.V)
  solve_linear!(get_free_dof_values(s.w), s.op, s.b, ctx)   # symmetric matrix: Aᵀ = A
  w, dΩ = s.w, s.measure
  δ -> ∫(w * δ)dΩ
end

"""
    Heaviside(; β, η)

Smooth Heaviside projection (Wang, Lazarov and Sigmund, 2011), applied point by point at the quadrature points:

    H(ρ) = (tanh(βη) + tanh(β(ρ - η))) / (tanh(βη) + tanh(β(1 - η)))

`β > 0` (sharpness) and `0 < η < 1` (threshold) are mandatory. `β` can be changed during the optimisation with
`set_beta!` (continuation); every field and form built from this object sees the new value, and the physical fields
that use it count as changed.
"""
mutable struct Heaviside <: FieldMap
  β::Float64
  η::Float64
  version::Int
  function Heaviside(β::Real, η::Real)
    β > 0 || throw(ArgumentError("Heaviside: β must be positive"))
    0 < η < 1 || throw(ArgumentError("Heaviside: η must be in (0, 1)"))
    new(β, η, 0)
  end
end
Heaviside(; β, η) = Heaviside(β, η)

"""
    set_beta!(h::Heaviside, β)

Changes the sharpness of the projection.
"""
function set_beta!(h::Heaviside, β)
  β > 0 || throw(ArgumentError("Heaviside: β must be positive"))
  h.β = β
  h.version += 1
  h
end

struct HeavisideValue <: Function
  h::Heaviside
end
struct HeavisideSlope <: Function
  h::Heaviside
end

function (f::HeavisideValue)(ρ)
  β, η = f.h.β, f.h.η
  (tanh(β * η) + tanh(β * (ρ - η))) / (tanh(β * η) + tanh(β * (1 - η)))
end

function (f::HeavisideSlope)(ρ)
  β, η = f.h.β, f.h.η
  β * (1 - tanh(β * (ρ - η))^2) / (tanh(β * η) + tanh(β * (1 - η)))
end

struct HeavisideStage{I,F,D}
  h::Heaviside
  input::I
  out::F
  slope::D
end

_stage(h::Heaviside, input) = HeavisideStage(h, input, Operation(HeavisideValue(h))(input), Operation(HeavisideSlope(h))(input))
output(s::HeavisideStage) = s.out
parameter_version(s::HeavisideStage) = s.h.version
update_stage!(s::HeavisideStage, ctx) = s
function pullback(s::HeavisideStage, L, ctx)
  slope = s.slope
  δ -> L(slope * δ)
end
