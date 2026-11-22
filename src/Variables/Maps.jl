# Maps between levels. A map is a specification; applied to an input field it creates a stage that owns its output
# (a fixed object updated in place, so weak forms can capture it) and knows how to pull a sensitivity back.
#
# Stage interface:
#   output(s)                 the output field (FEFunction or CellField)
#   update_stage!(s, ctx)     recompute the output from the input (in place)
#   pullback(s, L, ctx, key)  lazy stages: L: δ_out ↦ DomainContribution, linear form on the output; returns the
#                             linear form on the input (key: identifies the form for the compiled assemblers)
#   accumulate!(s, L, w, key) stages that are not lazy (filter, cell mean): assembles L and adds w times it to the
#                             accumulated vector of the stage; nothing is solved yet
#   flush!(s, ctx)            stages that are not lazy: pulls the accumulated vector back once (one solve of a filter)
#                             and returns the linear form on the input (or a DofFunctional, a vector on its free
#                             values: Ersatz); the accumulator is emptied
#   parameter_version(s)      counter of the parameters of the map (β of a TanhProjection, the Parameters of a
#                             PointwiseMap, the dependencies of a filter's operator); 0 if it has none
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

# The frozen zone of a design field, imposed at the end of every chain built from it: on the frozen cells the output
# is the image of the frozen value (θ = χ ? value : chain(x)); its derivative vanishes there. Affine, lazy, no solve.
# The image of the frozen value is kept as one value per cell (`img`, P0), recomputed from its lazy form only when a
# parameter of the chain changes (`_refresh_mask!`), so θ costs one selection per point and not a second projection.
struct MaskStage{I,O,F,C,V,L,R}
  input::I
  out::O
  free::F
  χ::C
  img::V            # P0: image of the frozen value on the frozen cells
  img_lazy::L       # its lazy form (the pointwise maps applied to the frozen value)
  reference::R      # the pointwise maps applied to the design field x (for check_consistency)
  seen::Base.RefValue{Int}   # parameter version of the chain at the last refresh
end
_select_frozen(c, χ, v) = χ > 0.5 ? v : c
function _mask_stage(input, zone, img_lazy, reference)
  img = FEFunction(get_fe_space(zone.value), zeros(num_free_dofs(get_fe_space(zone.value))))
  s = MaskStage(input, Operation(_select_frozen)(input, zone.χ, img), zone.free, zone.χ, img, img_lazy, reference, Ref(-1))
  _refresh_mask!(s, 0)
end
function _refresh_mask!(s::MaskStage, version)
  s.seen[] == version && return s
  interpolate!(s.img_lazy, get_free_dof_values(s.img), get_fe_space(s.img))   # exact: the value is constant per cell
  s.seen[] = version
  s
end
output(s::MaskStage) = s.out
update_stage!(s::MaskStage, ctx) = s
pullback(s::MaskStage, L, ctx, key) = (free = s.free; δ -> L(free * δ))
is_linear_stage(::MaskStage) = true

# maps evaluated point by point (they carry the frozen value into θ); filters and cell means leave it as it is. A new
# kind of FieldMap that acts point by point must declare `_pointwise(::MyMap) = true`; one that carries the frozen value
# through another map declares `_frozen_map(::MyMap) = that map` (Ersatz: the step of the value).
_pointwise(m) = false

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
  acc::Vector{Float64}             # accumulated sensitivities on the output (reverse sweep)
  pending::Base.RefValue{Bool}     # something was accumulated since the last flush
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
  FilterStage(f.op, input, f.measure, out, w, b, zero(b), Ref(false), V, form_assembler(f.backend, U, V))
end

output(s::FilterStage) = s.out

function update_stage!(s::FilterStage, ctx)
  dΩ, xin = s.measure, s.input
  assemble_form!(s.b, s.assem, v -> ∫(xin * v)dΩ, :forward)
  solve_linear!(get_free_dof_values(s.out), s.op, s.b, ctx)
  s
end

function accumulate!(s::FilterStage, L, w::Float64, key)
  assemble_form!(s.b, s.assem, L, key)
  s.acc .+= w .* s.b
  s.pending[] = true
  s
end

function flush!(s::FilterStage, ctx)
  solve_linear!(get_free_dof_values(s.w), s.op, s.acc, ctx)   # symmetric matrix: Aᵀ = A
  fill!(s.acc, 0.0)
  s.pending[] = false
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
# the dependencies of the filter's operator (a Parameter read by its form): a change re-solves the filter
parameter_version(s::FilterStage) = sum(state_version, s.op.depends; init = 0)
has_parameters(s::FilterStage) = !isempty(s.op.depends)
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
`state_version`), list them in `parameters`, so the physical fields count as changed when they change (continuation). The value of a frozen zone through the map is also recomputed only then.
Lazy, like [`TanhProjection`](@ref): nothing is recomputed, the forms evaluate `f` and `df` when they are assembled.

```julia
const p = Parameter(8.0)       # const: closures that read it do not allocate
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
PointwiseMap(f, df; parameters = ()) =
  PointwiseMap(f, df, parameters isa Union{Tuple,AbstractVector} ? Tuple(parameters) : (parameters,))

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
  acc::Vector{Float64}
  pending::Base.RefValue{Bool}
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
  CellMeanStage(input, dΩ, FEFunction(V0, zeros(n)), FEFunction(V0, zeros(n)), zeros(n), zeros(n), Ref(false), vol, V0,
                form_assembler(c.backend, V0, V0))
end

output(s::CellMeanStage) = s.out

function update_stage!(s::CellMeanStage, ctx)
  dΩ, xin = s.measure, s.input
  assemble_form!(s.b, s.assem, v -> ∫(xin * v)dΩ, :forward)
  get_free_dof_values(s.out) .= s.b ./ s.vol
  s
end

function accumulate!(s::CellMeanStage, L, w::Float64, key)
  assemble_form!(s.b, s.assem, L, key)          # L applied to the P0 basis: the derivative per cell
  s.acc .+= w .* s.b
  s.pending[] = true
  s
end

function flush!(s::CellMeanStage, ctx)
  get_free_dof_values(s.w) .= s.acc ./ s.vol
  fill!(s.acc, 0.0)
  s.pending[] = false
  w, dΩ = s.w, s.measure
  δ -> ∫(w * δ)dΩ
end

# accumulators of the stages that are not lazy (reverse sweep)
is_pending(s) = false
is_pending(s::Union{FilterStage,CellMeanStage}) = s.pending[]
reset_accumulator!(s) = s
reset_accumulator!(s::Union{FilterStage,CellMeanStage}) = (fill!(s.acc, 0.0); s.pending[] = false; s)

is_lazy(::CellMeanStage) = false
is_linear_stage(::CellMeanStage) = true

Base.show(io::IO, ::Identity) = print(io, "Identity()")
Base.show(io::IO, f::HelmholtzFilter) = print(io, "HelmholtzFilter(", f.backend, ")")
Base.show(io::IO, h::TanhProjection) = print(io, "TanhProjection(β = ", h.β, ", η = ", h.η, ")")
Base.show(io::IO, c::CellMean) = print(io, "CellMean(", c.backend, ")")
Base.show(io::IO, c::ComposedMap) = join(io, reverse(c.maps), " ∘ ")
Base.show(io::IO, m::PointwiseMap) = print(io, "PointwiseMap(", length(m.parameters), " parameters)")

_pointwise(::Identity) = true
_pointwise(::TanhProjection) = true
_pointwise(::PointwiseMap) = true

# whether the stage has a derivative with respect to its input (Step() has none)
is_differentiable(s) = true

# the map that carries the value of a frozen zone through a map of the chain (nothing: the value stays as it is)
_frozen_map(m) = _pointwise(m) ? m : nothing

# ---------------------------------------------------------------------------------------------------------------
# Interface maps: from a level-set-like field φ (φ ≥ 0 solid) to the material

"""
    Step()

Interface map: the material is the Heaviside step of the input at each quadrature point, `χ = 1` where `φ ≥ 0` and
`0` elsewhere (the abrupt step of the ersatz material of Li et al. 2023). The material of a cell cut by the interface
depends on which quadrature points fall on each side: integrate with a higher degree to smooth the jumps (see
[`design_measures`](@ref)). It has no derivative with respect to φ: a method that differentiates through the maps
(`DensityMethod`) refuses it when the run starts, and [`check_consistency`](@ref HyperShape.Checks.check_consistency)
reports the derivative checks as not applicable; use [`Ersatz`](@ref)`()` to check the derivatives of the forms (they
are the same). Lazy, point by point (a frozen zone `φ = 1` gives `χ = 1`). The refusal looks at the whole
`PhysicalFields`: a field through `Step()` next to differentiable ones goes in a `PhysicalFields` of its own.
"""
struct Step <: FieldMap end

struct StepValue <: Function end
(::StepValue)(φ) = ifelse(φ >= 0, one(φ), zero(φ))

struct StepStage{I,F}
  input::I
  out::F
end
_stage(::Step, input) = StepStage(input, Operation(StepValue())(input))
output(s::StepStage) = s.out
update_stage!(s::StepStage, ctx) = s
pullback(s::StepStage, L, ctx, key) =
  throw(ArgumentError("Step(): the step has no derivative with respect to the design (use Ersatz(), or a method that " *
                      "takes the sensitivity with respect to the physical field)"))
is_differentiable(::StepStage) = false
_pointwise(::Step) = true
Base.show(io::IO, ::Step) = print(io, "Step()")

"""
    Ersatz(; backend = :compiled)

Interface map of the classical ersatz material (Allaire, Jouve and Toader, 2004): the material of each cell is the
fraction of the cell where the input `φ` is ≥ 0, `χ_K = |{φ_h ≥ 0} ∩ K| / |K|`, a cell-wise (P0) field. Exact for the
linear interpolant of φ on simplices; quadrilaterals and hexahedra are split into 4 triangles and 24 tetrahedra
around their centre (whose value is the mean of the nodes), so the fraction is exact for that subdivision (not for the
bilinear interpolant: up to about 0.08 on a cell with values in a saddle) and O(h²) with respect to a smooth interface. Continuous in φ, with its exact derivative (closed forms, no
cancellation). The input must be a nodal P1 scalar field: the design field or the output of a filter (`Ersatz() ∘
HelmholtzFilter(…)`); triangles, quadrilaterals, tetrahedra and hexahedra. A volume of the material is exact with any
quadrature.

The derivative is nonzero only on the cells cut by the interface: a design without interface (uniform φ, or φ at a
bound everywhere) does not move, and holes are not nucleated by this derivative; `DensityMethod` refuses to start
from a design without cut cells. A cell with `φ ≡ 0` counts as material (derivative zero). With a frozen zone `φ = 1`
the cells next to it are partly material too (its nodes on their boundary have φ = 1). Cost (one thread, Mac M5 Max):
about 12 ns per quadrilateral and 7 ns per tetrahedron to update or pull back; a hexahedron cut by the interface about
350 ns (24 tetrahedra), one without interface a few ns.
`backend` assembles the forms on the cell-wise basis (`:compiled`, HyperFEMKernels, or `:lazy`, Gridap). Not lazy:
the fractions are computed when the design changes (a loop over the cells, no solve).

```julia
θ  = PhysicalFields(x; ρ = :φ => Ersatz() ∘ HelmholtzFilter(dΩ; rmin, ls = LUSolver()))
dΩ = design_measures(θ; degree = 2)
```
"""
struct Ersatz <: FieldMap
  backend::Symbol
  Ersatz(backend::Symbol) = new(_check_backend(backend))
end
Ersatz(; backend::Symbol = :compiled) = Ersatz(backend)
Base.show(io::IO, e::Ersatz) = print(io, "Ersatz(", e.backend, ")")
_frozen_map(::Ersatz) = Step()        # the material of a frozen zone is the step of its value

# A linear functional on the free values of the input of a stage, given by its vector (what Ersatz pulls back: the
# derivative is a sum over cells of nodal values, not an integral form)
struct DofFunctional
  v::Vector{Float64}
end

# K: :tri, :quad, :tet or :hex; N: nodes per cell; W: weights of the subdivision per cell (nothing for simplices)
struct ErsatzStage{K,N,I,F,U,W,V,A}
  input::I
  out::F
  U::U
  data::Vector{Int32}           # cell dof ids of the input (flat), positive: free, negative: Dirichlet
  ptrs::Vector{Int32}
  cdof::Vector{Int32}           # the dof of each cell in the cell-wise space
  w::W
  b::Vector{Float64}
  acc::Vector{Float64}
  pending::Base.RefValue{Bool}
  gin::Vector{Float64}
  V0::V
  assem::A
end

function _stage(e::Ersatz, input)
  ok = input isa Gridap.FESpaces.SingleFieldFEFunction
  U = ok ? get_fe_space(input) : nothing
  Ω = ok ? get_triangulation(U) : nothing
  ids = ok ? Gridap.Arrays.Table(get_cell_dof_ids(U)) : nothing
  D = ok ? num_cell_dims(Ω) : 0
  N = ok && !isempty(ids) ? length(ids[1]) : 0
  K = (D, N) == (2, 3) ? :tri : (D, N) == (2, 4) ? :quad : (D, N) == (3, 4) ? :tet : (D, N) == (3, 8) ? :hex : nothing
  (ok && K !== nothing && all(c -> length(ids[c]) == N, 1:length(ids)) && num_vertices_per_cell(Ω) == N) ||
    throw(ArgumentError("Ersatz: the input must be a nodal P1 scalar field on triangles, quadrilaterals, tetrahedra " *
                        "or hexahedra (the design field or the output of a filter), not a lazy field, a P0 or a P2 field"))
  V0 = FESpace(Ω, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
  n = num_free_dofs(V0)
  cdof = Int32[c[1] for c in get_cell_dof_ids(V0)]
  X = get_cell_coordinates(Ω)
  w = K === :quad ? [_quad_weights(X[c]) for c in 1:length(X)] :
      K === :hex ? [_hex_weights(X[c]) for c in 1:length(X)] : nothing
  out = FEFunction(V0, zeros(n))
  assem = form_assembler(e.backend, V0, V0)
  ErsatzStage{K,N,typeof(input),typeof(out),typeof(U),typeof(w),typeof(V0),typeof(assem)}(input, out, U,
    Vector{Int32}(ids.data), Vector{Int32}(ids.ptrs), cdof, w, zeros(n), zeros(n), Ref(false),
    zeros(num_free_dofs(U)), V0, assem)
end

num_vertices_per_cell(Ω) = length(first(get_cell_coordinates(Ω)))

# the nodal values of cell c as a tuple (written out: a closure over the stage would allocate per cell)
@inline _node_value(data, fv, dv, j) = (d = data[j]; d > 0 ? fv[d] : dv[-d])
@inline function _cell_values(s::ErsatzStage{K,3}, fv, dv, c) where {K}
  p, a = s.ptrs[c], s.data
  (_node_value(a, fv, dv, p), _node_value(a, fv, dv, p + 1), _node_value(a, fv, dv, p + 2))
end
@inline function _cell_values(s::ErsatzStage{K,4}, fv, dv, c) where {K}
  p, a = s.ptrs[c], s.data
  (_node_value(a, fv, dv, p), _node_value(a, fv, dv, p + 1), _node_value(a, fv, dv, p + 2), _node_value(a, fv, dv, p + 3))
end
@inline function _cell_values(s::ErsatzStage{K,8}, fv, dv, c) where {K}
  p, a = s.ptrs[c], s.data
  (_node_value(a, fv, dv, p), _node_value(a, fv, dv, p + 1), _node_value(a, fv, dv, p + 2), _node_value(a, fv, dv, p + 3),
   _node_value(a, fv, dv, p + 4), _node_value(a, fv, dv, p + 5), _node_value(a, fv, dv, p + 6), _node_value(a, fv, dv, p + 7))
end
@inline _fraction(s::ErsatzStage{:tri}, φ, c) = cell_fraction(φ)
@inline _fraction(s::ErsatzStage{:tet}, φ, c) = cell_fraction(φ)
@inline _fraction(s::ErsatzStage{:quad}, φ, c) = quad_fraction(φ, s.w[c])
@inline _fraction(s::ErsatzStage{:hex}, φ, c) = hex_fraction(φ, s.w[c])

output(s::ErsatzStage) = s.out

function update_stage!(s::ErsatzStage, ctx)
  _fractions!(get_free_dof_values(s.out), s, get_free_dof_values(s.input), get_dirichlet_dof_values(s.U))
  s
end
function _fractions!(χ, s, fv, dv)
  @inbounds for c in eachindex(s.cdof)
    v = _cell_values(s, fv, dv, c)
    isfinite(sum(v)) || throw(ArgumentError("Ersatz: the input field is not finite on cell $c"))
    χ[s.cdof[c]] = _fraction(s, v, c)[1]
  end
  χ
end

# whether some cell is cut by the interface (0 < χ < 1)
has_interface(s::ErsatzStage) = any(χ -> 0 < χ < 1, get_free_dof_values(s.out))

function accumulate!(s::ErsatzStage, L, w::Float64, key)
  assemble_form!(s.b, s.assem, L, key)          # L applied to the cell-wise basis: the derivative per cell
  s.acc .+= w .* s.b
  s.pending[] = true
  s
end

function flush!(s::ErsatzStage, ctx)
  _pull_fractions!(s.gin, s, get_free_dof_values(s.input), get_dirichlet_dof_values(s.U))
  fill!(s.acc, 0.0)
  s.pending[] = false
  DofFunctional(s.gin)
end
function _pull_fractions!(g, s::ErsatzStage{K,N}, fv, dv) where {K,N}
  fill!(g, 0.0)
  @inbounds for c in eachindex(s.cdof)
    a = s.acc[s.cdof[c]]
    a == 0 && continue
    _, dχ = _fraction(s, _cell_values(s, fv, dv, c), c)
    p = s.ptrs[c]
    for i in 1:N
      d = s.data[p+i-1]
      d > 0 && (g[d] += a * dχ[i])        # frozen (Dirichlet) values are not variables
    end
  end
  g
end

is_lazy(::ErsatzStage) = false
is_pending(s::ErsatzStage) = s.pending[]
reset_accumulator!(s::ErsatzStage) = (fill!(s.acc, 0.0); s.pending[] = false; s)

# the stages that receive what Ersatz pulls back: a filter (its accumulator is on the free values of its output, the
# input of Ersatz) and the design field (see PhysicalFields)
function accumulate!(s::FilterStage, L::DofFunctional, w::Float64, key)
  s.acc .+= w .* L.v
  s.pending[] = true
  s
end
