# Maps between levels. A map is a specification; applied to an input field it creates a stage that owns its output
# (a fixed object updated in place, so weak forms can capture it) and knows how to pull a sensitivity back.
#
# Stage interface:
#   output(s)                 the output field (FEFunction or CellField)
#   update_stage!(s, ctx)     recompute the output from the input (in place)
#   pullback(s, L, ctx, key)  L: δ_out ↦ DomainContribution, linear form on the output; returns the linear form on the input
#                             (key: identifies the form for the compiled assemblers, see Assembly.jl)
#   parameter_version(s)      counter of the parameters of the map (β of a TanhProjection); 0 if it has none
#   is_lazy(s)                true if the output is evaluated lazily (nothing to recompute when an earlier parameter changes)

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
pullback(s::IdentityStage, L, ctx, key) = L
parameter_version(s) = 0

# whether the stage is linear in its input (a nonlinear stage makes a volume a nonlinear constraint)
is_linear_stage(s) = false

"""
    HelmholtzFilter(dΩ; rmin, order = 1, backend = :compiled, model_backend = <default>, solve = <default>, kwargs...)
    HelmholtzFilter(op::LinearOperator, dΩ; backend = :compiled)

PDE filter (Lazarov and Sigmund, 2011): the filtered field ρ̃ solves `a(ρ̃, v) = ∫(x v) dΩ` for all `v`.

- **Short form** (the usual filter): `a(ρ̃, v) = ∫(r² ∇ρ̃⋅∇v + ρ̃ v)dΩ` with `r = rmin / (2√3)`, homogeneous Neumann
  conditions, on a Lagrangian space of degree `order` (continuous) over the triangulation of `dΩ`. `kwargs` go to
  HyperFEM's `StaticLinearModel` (for example `ls = LUSolver()`, `ls = CGSolver(…)` or a PETSc solver; without `ls`
  HyperFEM's default solver is used), `model_backend` is passed to it as its `backend` (who assembles the matrix;
  HyperFEM's default if omitted) and `solve` goes to the `LinearOperator` (default: assemble when the matrix changes).
- **General form**: `op` is a [`LinearOperator`](@ref) whose model carries the bilinear form and the solver written
  by the user (Robin terms, Dirichlet conditions, anisotropy…). The bilinear form must be symmetric (the same solve
  gives the transpose for the sensitivities; checked by `check_consistency`) and the trial space may only have
  homogeneous Dirichlet values (checked).

The output is a field of the trial space of the filter (usually P1), also for a P0 design field. `backend` selects who
assembles the load vectors of the filter: `:compiled` (default, HyperFEMKernels) or `:lazy` (Gridap); it is
independent of the backend of the filter model, which assembles the matrix.

Cost: one assembly of a load vector and one solve each time the design changes, and the same for each sensitivity
pulled back.
"""
struct HelmholtzFilter{O<:LinearOperator,M} <: FieldMap
  op::O
  measure::M
  backend::Symbol
  HelmholtzFilter(op::O, dΩ::M, backend::Symbol) where {O<:LinearOperator,M<:Gridap.CellData.Measure} =
    new{O,M}(op, dΩ, _check_backend(backend))
end
HelmholtzFilter(op::LinearOperator, dΩ; backend::Symbol = :compiled) = HelmholtzFilter(op, dΩ, backend)

function HelmholtzFilter(dΩ::Gridap.CellData.Measure; rmin::Real, order::Integer = 1, backend::Symbol = :compiled,
                         model_backend = nothing, solve = nothing, kwargs...)
  rmin > 0 || throw(ArgumentError("HelmholtzFilter: rmin must be positive"))
  order >= 1 || throw(ArgumentError("HelmholtzFilter: order must be at least 1 (a continuous space)"))
  Ω = get_triangulation(dΩ.quad)
  V = TestFESpace(Ω, ReferenceFE(lagrangian, Float64, order); conformity = :H1)
  U = TrialFESpace(V)
  r2 = (rmin / (2sqrt(3)))^2
  a(u, v) = ∫(r2 * ∇(u) ⋅ ∇(v) + u * v)dΩ
  model = model_backend === nothing ? StaticLinearModel(a, U, V, NothingBC(); kwargs...) :
                                      StaticLinearModel(a, U, V, NothingBC(); backend = model_backend, kwargs...)
  op = solve === nothing ? LinearOperator(model) : LinearOperator(model; solve)
  HelmholtzFilter(op, dΩ, backend)
end

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
  # the load vector carries no Dirichlet lift: only homogeneous Dirichlet values are allowed
  all(iszero, get_dirichlet_dof_values(U)) ||
    throw(ArgumentError("HelmholtzFilter: the trial space of the filter model has non-zero Dirichlet values"))
  out = FEFunction(U, zeros(num_free_dofs(U)))
  w = FEFunction(U, zeros(num_free_dofs(U)))
  b = zeros(num_free_dofs(V))
  FilterStage(f.op, input, f.measure, out, w, b, V, form_assembler(f.backend, U, V))
end

output(s::FilterStage) = s.out

function update_stage!(s::FilterStage, ctx)
  dΩ, xin = s.measure, s.input
  assemble_form!(s.b, s.assem, v -> ∫(xin * v)dΩ, :forward)
  solve_linear!(get_free_dof_values(s.out), s.op, s.b, ctx)
  s
end

function pullback(s::FilterStage, L, ctx, key)
  assemble_form!(s.b, s.assem, L, key)
  solve_linear!(get_free_dof_values(s.w), s.op, s.b, ctx)   # symmetric matrix: Aᵀ = A
  w, dΩ = s.w, s.measure
  δ -> ∫(w * δ)dΩ
end

"""
    TanhProjection(; β, η)

Smooth approximation of the Heaviside step (Wang, Lazarov and Sigmund, 2011), applied point by point at the quadrature points:

    H(ρ) = (tanh(βη) + tanh(β(ρ - η))) / (tanh(βη) + tanh(β(1 - η)))

`β > 0` (sharpness) and `0 < η < 1` (threshold) are mandatory. `β` can be changed during the optimisation with
`set_beta!` (continuation); every field and form built from this object sees the new value, and the physical fields
that use it count as changed.
"""
mutable struct TanhProjection <: FieldMap
  β::Float64
  η::Float64
  version::Int
  function TanhProjection(β::Real, η::Real)
    β > 0 || throw(ArgumentError("TanhProjection: β must be positive"))
    0 < η < 1 || throw(ArgumentError("TanhProjection: η must be in (0, 1)"))
    new(β, η, 0)
  end
end
TanhProjection(; β, η) = TanhProjection(β, η)

"""
    set_beta!(h::TanhProjection, β)

Changes the sharpness of the projection.
"""
function set_beta!(h::TanhProjection, β)
  β > 0 || throw(ArgumentError("TanhProjection: β must be positive"))
  β == h.β && return h
  setfield!(h, :β, Float64(β))
  setfield!(h, :version, h.version + 1)
  h
end

Base.setproperty!(::TanhProjection, s::Symbol, v) =
  throw(ArgumentError("TanhProjection: use set_beta! to change β; η and the other fields cannot be changed"))

struct TanhValue <: Function
  h::TanhProjection
end
struct TanhSlope <: Function
  h::TanhProjection
end

function (f::TanhValue)(ρ)
  β, η = f.h.β, f.h.η
  (tanh(β * η) + tanh(β * (ρ - η))) / (tanh(β * η) + tanh(β * (1 - η)))
end

function (f::TanhSlope)(ρ)
  β, η = f.h.β, f.h.η
  β * (1 - tanh(β * (ρ - η))^2) / (tanh(β * η) + tanh(β * (1 - η)))
end

struct TanhStage{I,F,D}
  h::TanhProjection
  input::I
  out::F
  slope::D
end

_stage(h::TanhProjection, input) = TanhStage(h, input, Operation(TanhValue(h))(input), Operation(TanhSlope(h))(input))
output(s::TanhStage) = s.out
parameter_version(s::TanhStage) = s.h.version
has_parameters(s) = false
has_parameters(s::TanhStage) = true
is_lazy(s) = true
is_lazy(s::FilterStage) = false
update_stage!(s::TanhStage, ctx) = s
function pullback(s::TanhStage, L, ctx, key)
  slope = s.slope
  δ -> L(slope * δ)
end

"""
    PointwiseMap(f, df; parameters = ())

Map applied point by point at the quadrature points, `θ = f(input)`, with its derivative `df` written by the user
(no automatic differentiation; check it with `check_consistency`): for example a projection or a threshold of the
user's own. `f` and `df` are functions of one number. If they read [`Parameter`](@ref HyperShape.Operators.Parameter)s (or other objects with a
`state_version`), list them in `parameters`, so the physical fields count as changed when they change (continuation).
Lazy, like [`TanhProjection`](@ref): nothing is recomputed, the forms evaluate `f` and `df` when they are assembled.

```julia
p = Parameter(8.0)
smooth_step = PointwiseMap(ρ -> 1 / (1 + exp(-p[] * (ρ - 0.5))),
                           ρ -> (s = 1 / (1 + exp(-p[] * (ρ - 0.5))); p[] * s * (1 - s)); parameters = (p,))
θ = PhysicalFields(x; ρ = smooth_step ∘ HelmholtzFilter(dΩ; rmin, ls))
```
"""
struct PointwiseMap{F,D,P<:Tuple} <: FieldMap
  f::F
  df::D
  parameters::P
  function PointwiseMap(f::F, df::D, parameters::P) where {F,D,P<:Tuple}
    for q in parameters
      applicable(state_version, q) ||
        throw(ArgumentError("PointwiseMap: a parameter must implement state_version (a Parameter); got $(typeof(q))"))
    end
    new{F,D,P}(f, df, parameters)
  end
end
PointwiseMap(f, df; parameters = ()) = PointwiseMap(f, df, Tuple(parameters))

struct PointwiseStage{M,I,F,D}
  map::M
  input::I
  out::F
  slope::D
end

_stage(m::PointwiseMap, input) = PointwiseStage(m, input, Operation(m.f)(input), Operation(m.df)(input))
output(s::PointwiseStage) = s.out
parameter_version(s::PointwiseStage) = sum(state_version, s.map.parameters; init = 0)
has_parameters(s::PointwiseStage) = !isempty(s.map.parameters)
update_stage!(s::PointwiseStage, ctx) = s
function pullback(s::PointwiseStage, L, ctx, key)
  slope = s.slope
  δ -> L(slope * δ)
end

is_linear_stage(::IdentityStage) = true
is_linear_stage(::FilterStage) = true

"""
    CellMean(dΩ; backend = :compiled)

Mean of the input field over each cell of the triangulation of `dΩ`: a P0 field, `x̄_K = ∫_K x dΩ / |K|` (with the
quadrature of `dΩ`). On a measure over part of the domain the field is the mean on those cells and its integrals over
the whole domain only see that part (Gridap's rule for fields on a sub-triangulation). After a filter it gives the density of codes such as top88 (`xPhys = TF'·K⁻¹·TF·x`): one value
per element instead of the filtered field at the quadrature points. Linear in its input (a volume after it stays a
linear constraint); no solve. `backend` assembles its vectors (`:compiled`, HyperFEMKernels, or `:lazy`, Gridap).

```julia
θ = PhysicalFields(x; ρ = CellMean(dΩ) ∘ HelmholtzFilter(dΩ; rmin, ls = LUSolver()))
```
"""
struct CellMean{M} <: FieldMap
  measure::M
  backend::Symbol
  CellMean(dΩ::M, backend::Symbol) where {M<:Gridap.CellData.Measure} = new{M}(dΩ, _check_backend(backend))
end
CellMean(dΩ::Gridap.CellData.Measure; backend::Symbol = :compiled) = CellMean(dΩ, backend)

struct CellMeanStage{I,M,F,W,V,A}
  input::I
  measure::M
  out::F          # P0 field of the cell means
  w::W            # adjoint field (pullback)
  b::Vector{Float64}
  vol::Vector{Float64}
  V0::V
  assem::A
end

function _stage(c::CellMean, input)
  dΩ = c.measure
  Ω = get_triangulation(dΩ.quad)
  V0 = FESpace(Ω, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
  n = num_free_dofs(V0)
  vol = collect(assemble_vector(v -> ∫(v)dΩ, V0))
  all(>(0), vol) || throw(ArgumentError("CellMean: a cell has zero measure"))
  CellMeanStage(input, dΩ, FEFunction(V0, zeros(n)), FEFunction(V0, zeros(n)), zeros(n), vol, V0,
                form_assembler(c.backend, V0, V0))
end

output(s::CellMeanStage) = s.out

function update_stage!(s::CellMeanStage, ctx)
  dΩ, xin = s.measure, s.input
  assemble_form!(s.b, s.assem, v -> ∫(xin * v)dΩ, :forward)
  get_free_dof_values(s.out) .= s.b ./ s.vol
  s
end

function pullback(s::CellMeanStage, L, ctx, key)
  assemble_form!(s.b, s.assem, L, key)          # L applied to the P0 basis: the derivative per cell
  get_free_dof_values(s.w) .= s.b ./ s.vol
  w, dΩ = s.w, s.measure
  δ -> ∫(w * δ)dΩ
end

is_lazy(::CellMeanStage) = false
is_linear_stage(::CellMeanStage) = true

Base.show(io::IO, ::Identity) = print(io, "Identity()")
Base.show(io::IO, f::HelmholtzFilter) = print(io, "HelmholtzFilter(", f.backend, ")")
Base.show(io::IO, h::TanhProjection) = print(io, "TanhProjection(β = ", h.β, ", η = ", h.η, ")")
Base.show(io::IO, c::CellMean) = print(io, "CellMean(", c.backend, ")")
Base.show(io::IO, c::ComposedMap) = join(io, reverse(c.maps), " ∘ ")
Base.show(io::IO, m::PointwiseMap) = print(io, "PointwiseMap(", length(m.parameters), " parameters)")
