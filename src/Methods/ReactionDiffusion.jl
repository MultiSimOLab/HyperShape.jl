# Reaction–diffusion method: an implicit step of a reaction–diffusion equation for a level set (or a density), with
# the volume met exactly at every step by a scalar multiplier (Choi et al. 2011, Yamada et al. 2010, Otomori et al.
# 2015, Kim et al. 2020, Cui et al. 2023 for the exact multiplier).

# ---------------------------------------------------------------------------------------------------------------
# Normalisation of the sensitivity

"""
    L1Mean(c = 1.0)

Normalisation of the sensitivity of [`ReactionDiffusion`](@ref) (its default): `C = c |D| / Σᵢ |gᵢ|`, with `gᵢ` the
derivative of the objective with respect to the nodal value `i` (a dual vector) and `|D|` the measure of the design
domain. The normalised sensitivity density `C gᵢ / mᵢ` (`mᵢ` the lumped mass) then has mean absolute value `c` over
the domain, whatever the mesh. `c` scales the reaction against the diffusion `τ`.
"""
struct L1Mean
  c::Float64
  function L1Mean(c::Real = 1.0)
    (isfinite(c) && c > 0) || throw(ArgumentError("L1Mean: c must be positive and finite, got $c"))
    new(Float64(c))
  end
end

"""
    L2Norm()

Normalisation of the sensitivity of [`ReactionDiffusion`](@ref): `C = |D| / (Σᵢ gᵢ² / mᵢ)^½`, the normalisation of
Kim et al. (2020) with the measure `|D|` of the domain in place of `|D|^½` (so that `τ` keeps the same meaning as with
[`L1Mean`](@ref)).
"""
struct L2Norm end

"""
    MaxAbs()

Normalisation of the sensitivity of [`ReactionDiffusion`](@ref): `C = 1 / maxᵢ |gᵢ / mᵢ|` (the largest normalised
sensitivity density is one). It depends on the mesh: a singular sensitivity (at a point load) grows when the mesh is
refined.
"""
struct MaxAbs end

# no allocation: reductions over lazy broadcasts
(n::L1Mean)(g, m, area) = n.c * area / sum(abs, g)
(::L2Norm)(g, m, area) = area / sqrt(sum(Broadcast.instantiate(Broadcast.broadcasted((a, b) -> a^2 / b, g, m))))
(::MaxAbs)(g, m, area) = 1 / maximum(Broadcast.instantiate(Broadcast.broadcasted((a, b) -> abs(a) / b, g, m)))

Base.show(io::IO, n::L1Mean) = print(io, "L1Mean(", _g(n.c), ")")
Base.show(io::IO, ::L2Norm) = print(io, "L2Norm()")
Base.show(io::IO, ::MaxAbs) = print(io, "MaxAbs()")

# The normalisation of Takezawa et al. (2010), `C = 1 / (Σᵢ gᵢ² / mᵢ)^½` (the L² norm without |D|), for the preset
# :takezawa2010 (internal: a function (g, m, area) -> C of the user replaces it)
struct _TakezawaNorm end
(::_TakezawaNorm)(g, m, area) = 1 / sqrt(sum(Broadcast.instantiate(Broadcast.broadcasted((a, b) -> a^2 / b, g, m))))
Base.show(io::IO, ::_TakezawaNorm) = print(io, "L² norm without |D| (Takezawa et al. 2010)")

# ---------------------------------------------------------------------------------------------------------------
# Exact volume

"""
    ExactVolume(; max_step = :auto, tol = 1e-4, maxiter = 50, constraint = nothing)

How [`ReactionDiffusion`](@ref) treats the volume (its default): natively, with a scalar multiplier `λ` found at
every step so that the clipped design `z = clamp(φ* − λ φ̌, lb, ub)` meets the constraint (`φ*` the design after the
reaction–diffusion step, `φ̌ = A⁻¹ m` the response of the step to a uniform unit source). It takes the constraint
built with [`Volume`](@ref HyperShape.Functionals.Volume) (`g ≤ 0`), wherever it is in the problem; with none or with
several, `solve!` throws: name the one to take with `constraint = :name` (any inequality `g ≤ 0` evaluated without a
state, linear or not). The other constraints go to the augmented Lagrangian of the method.

- `max_step`: how much the target may go down per iteration, as a fraction of the measure `|D|` of the design
  domain: the target of iteration `k` is `max(V_obj, V_{k−1} − max_step |D|)`, met with equality while it goes down;
  at `V_obj` the constraint is `g ≤ 0` with `λ ≥ 0`. `:auto` (default): `0.02` when the constraint reads the
  material through an interface map (`Step()`, `Ersatz()`), `nothing` otherwise (a density: the target from the
  first iteration). `nothing`: the target from the first iteration. A number in `(0, 1]`.
- `tol`: tolerance of the root as a fraction of `|D|`, like `max_step` (`1e-4`: the volume within `1e-4 |D|` of its
  target), converted to the units of the scaled constraint with its scale; the loop converges only with the
  constraint met to it.
- `maxiter`: evaluations of the constraint per step at most.

The root of `G(λ) = g(clamp(φ* − λ φ̌)) − g_k` (non-increasing in λ) starts from the previous `λ`, inside the bracket
given by the bounds (all values at the upper or at the lower bound): a linear constraint (a density) by Newton on the
clipped vector (only vectors, no solve; the algorithm of Cui et al. 2023); a nonlinear one with a derivative
(`Ersatz()`) by Newton with the derivative of the constraint, guarded by the bracket; `Step()` by regula falsi
(Illinois) on the volume computed by the material meter of the step (the same quadrature as the `Volume`, from the
nodal values: exact, cheap and without allocations), until at most one quadrature point changes side inside the
bracket; it ends on its feasible side (the volume of `Step()` is piecewise constant, and on coarse meshes the jump of
one point can be larger than `tol`). Without the meter (a `Volume` over part of the domain, another constraint) the
evaluations of `Step()` go through the constraint, until the trials differ by less than 1e-9. If the evaluations run out before a feasible trial is seen, the step takes the bound of the bracket
where every value is at its lower bound. Each evaluation of a nonlinear constraint updates the maps and integrates
(no state is solved, but the integral reserves memory: about 70 kB per evaluation and 130 kB per derivative with
`Ersatz()`, independent of the mesh); a linear one costs vector operations only, without allocations.
"""
struct ExactVolume
  max_step::Union{Symbol,Nothing,Float64}
  tol::Float64
  maxiter::Int
  constraint::Union{Nothing,Symbol}
  function ExactVolume(; max_step = :auto, tol::Real = 1e-4, maxiter::Integer = 50, constraint::Union{Nothing,Symbol} = nothing)
    (max_step === :auto || max_step === nothing || (max_step isa Real && 0 < max_step <= 1)) ||
      throw(ArgumentError("ExactVolume: max_step must be :auto, nothing or a fraction of |D| in (0, 1], got $(repr(max_step))"))
    (isfinite(tol) && tol > 0) || throw(ArgumentError("ExactVolume: tol must be positive, got $tol"))
    maxiter >= 2 || throw(ArgumentError("ExactVolume: maxiter must be at least 2, got $maxiter"))
    new(max_step isa Real ? Float64(max_step) : max_step, Float64(tol), Int(maxiter), constraint)
  end
end

Base.show(io::IO, v::ExactVolume) =
  print(io, "ExactVolume(max_step = ", repr(v.max_step), ", tol = ", v.tol, ", maxiter = ", v.maxiter,
        v.constraint === nothing ? "" : ", constraint = $(repr(v.constraint))", ")")

# ---------------------------------------------------------------------------------------------------------------
# The double-well reaction

"""
    DoubleWell(; W = 0.25, η = 10.0, T = 25.0, cfl = :auto, scheme = :implicit, settle = true, substeps = nothing)

The double-well reaction of [`ReactionDiffusion`](@ref) (Takezawa, Nishiwaki and Kitamura 2010; Kim et al. 2020, loop
2), given as `ReactionDiffusion(; τ, reaction = DoubleWell())`. The design is a density `φ ∈ [lb, ub]` (normally
`bounds = (0.0, 1.0)`, read through `Identity()` and an interpolation in the weak forms) that evolves by

    ∂φ̂/∂t = τ ℓ² Δφ̂ − W w′(φ̂) − 30 w(φ̂) (η C g/m + λ),     w(φ̂) = φ̂² (1 − φ̂)²,   φ̂ = (φ − lb)/(ub − lb),

with `C` the normalisation of the method (`normalize`, [`L2Norm`](@ref)`()` by default with the well), `g/m` the
sensitivity density and `λ` the multiplier of the volume, found exactly at every substep ([`ExactVolume`](@ref)).
Both the sensitivity and the multiplier are weighted by `30 w(φ̂)`, which vanishes at `0` and `1`: the reaction only
moves the interface and holes are not nucleated, so the start needs one (built with
[`hole_seed`](@ref HyperShape.Variables.hole_seed)`(Vφ; bounds = (0.0, 1.0), …)`). The well keeps the design close to
0/1 with an interface of width `ξ = √(τ ℓ² / 2W)`. `τ` (that is `κ = τ ℓ²`) is a physical parameter, the same on
every mesh: on a coarse mesh the interface is thinner than a cell.

- `W`: height of the well (barrier `W/16`). `η`: gain of the sensitivity. Takezawa et al. use `W = 1/4` and `η = 20` with
  `g/‖g‖_L²`; `η = 10` with `L2Norm()` (`|D| g/‖g‖_L²`) is the same on their domain of measure 2.
- `T`: the time of the evolution per iteration (one forward solve, one sensitivity), split into
  `n = ⌈T / (cfl Δt_lim)⌉` substeps of `Δt = T/n`, with the bound of Gershgorin `Δt_lim = 2 / maxᵢ(τℓ² Σⱼ|Kᵢⱼ| / mᵢ)`
  of the explicit diffusion, from the assembled `K` and lumped mass `m` once at the start (`0.375 h²/(τℓ²)` on a
  uniform mesh of squares; it holds on any mesh). `T` is physical: the evolution per iteration does not depend on the
  mesh, and the number of substeps grows like `(h₀/h)²` (the pointwise reaction needs `√(τℓ² Δt) ≲ 0.7 h`). The
  method keeps `Δt = :auto`.
- `cfl`: the substep relative to `Δt_lim`; `:auto` is `4/3` with `:implicit` and `1/3` with `:explicit` (`h²/(8τℓ²)`,
  the step of Takezawa et al.).
- `scheme = :implicit` (default): each substep diffuses implicitly, `(m/Δt + τℓ²K) ψ = m φ/Δt` with the operator
  factorised once (the one of the step), and then applies the semi-implicit pointwise reaction of Takezawa et al.
  (their Eqs. 44–45) to `ψ`, with `λ` inside the tilt term, by a root on vectors. `scheme = :explicit`: the explicit
  substeps of Takezawa et al. (`ψ = φ − Δt τℓ² m⁻¹ K φ`, no linear system), the scheme of the preset `:takezawa2010`.
- `substeps = n` (an integer): `n` substeps of the `Δt` of the method per iteration, and `T` is ignored; `Δt = :auto`
  of the method is then `cfl Δt_lim` and a number replaces it (with `:explicit`, `Δt ≤ Δt_lim` is checked). With
  `substeps = nothing` (default) a number `Δt` of the method is refused.
- `settle`: with the substeps frozen on one sensitivity the interface overshoots and comes back, and a few per cent of
  the material changes phase at every iteration. `settle = true` goes to one substep (of the same `Δt`) per iteration
  once the constraints are met and the relative range of the objective over 10 iterations is below `1e-3`, or at 75 %
  of `maxiter` (an event says so); the default stop is only checked from the iteration after the settling (or from the
  start when `n = 1`). `settle = false` keeps the substeps and does not hold the default stop (give `stop` and
  `maxiter` yourself).
- The default stop with the well (`default_stop`): the constraints met, the material changed less than `1e-2` in each
  of the last 5 iterations and the relative range of the objective over the last 10 below `1e-3`
  (`StopCriteria(material_tol = 1e-2, objective_range = 1e-3, combine = :all)`; `2e-3` without the well: the interface
  of the well keeps a cycle of a few per mille of the material per iteration), after the settling. `solve!(…; stop)`
  replaces it (and is not held until the settling).

Mesh: with `τ` and `T` fixed, the compliance of the cantilever 2 × 1 (κ = 1e-5, `T = 25`, seed `:hex`) is the same
within about 3 % on 100 × 50, 200 × 100 and 400 × 200 (2.558, 2.493, 2.426, with 2, 5 and 20 substeps), but the
topology depends on the seed and on the mesh (the double well does not nucleate holes, and the seed is resolved
differently on each mesh). The bridge of Takezawa et al. (a point load) is not comparable across meshes: its
compliance grows like `log(1/h)`.

With the well, `sensitivity` is not mandatory (it is the exact derivative, `:design`; `:physical` is refused),
`normalize` defaults to `L2Norm()` and `ExactVolume(max_step = :auto)` lowers the target `1 %` of `|D|` per iteration
(spread over the substeps). Not with an interface map (`Ersatz()`, `Step()`). Per iteration: one forward solve, one
adjoint, `n` back-substitutions (`:implicit`; none with `:explicit`) and `n` roots on vectors, no factorisation.
"""
struct DoubleWell
  W::Float64
  η::Float64
  T::Float64
  substeps::Int           # 0: from T
  cfl::Float64            # NaN: :auto
  scheme::Symbol
  settle::Bool
  function DoubleWell(; W::Real = 0.25, η::Real = 10.0, T::Real = 25.0, cfl = :auto, scheme::Symbol = :implicit,
                      settle::Bool = true, substeps::Union{Nothing,Integer} = nothing)
    (isfinite(W) && W > 0) || throw(ArgumentError("DoubleWell: W must be positive and finite, got $W"))
    (isfinite(η) && η > 0) || throw(ArgumentError("DoubleWell: η must be positive and finite, got $η"))
    (isfinite(T) && T > 0) || throw(ArgumentError("DoubleWell: T must be positive and finite, got $T"))
    scheme in (:implicit, :explicit) || throw(ArgumentError("DoubleWell: scheme must be :implicit or :explicit, got $(repr(scheme))"))
    substeps === nothing || substeps >= 1 || throw(ArgumentError("DoubleWell: substeps must be nothing or at least 1, got $substeps"))
    (cfl === :auto || (cfl isa Real && isfinite(cfl) && cfl > 0)) ||
      throw(ArgumentError("DoubleWell: cfl must be :auto or a positive number, got $(repr(cfl))"))
    scheme === :explicit && cfl isa Real && cfl > 1 &&
      throw(ArgumentError("DoubleWell: with scheme = :explicit cfl must be ≤ 1 (the bound of stability), got $cfl"))
    new(Float64(W), Float64(η), Float64(T), substeps === nothing ? 0 : Int(substeps), cfl === :auto ? NaN : Float64(cfl),
        scheme, settle)
  end
end

# the factor of Δt_lim with Δt = :auto
_cfl(d::DoubleWell) = isnan(d.cfl) ? (d.scheme === :explicit ? 1 / 3 : 4 / 3) : d.cfl

Base.show(io::IO, d::DoubleWell) =
  print(io, "DoubleWell(W = ", _g(d.W), ", η = ", _g(d.η), d.substeps == 0 ? ", T = $(_g(d.T))" : ", substeps = $(d.substeps)",
        ", cfl = ", isnan(d.cfl) ? ":auto" : _g(d.cfl), ", scheme = ", repr(d.scheme), ", settle = ", d.settle, ")")

# the substeps of an iteration from T: n = ⌈T/(cfl Δt_lim)⌉ and Δt = T/n (the relative tolerance keeps n when
# T/(cfl Δt_lim) is an integer up to rounding); with substeps given, those
_dw_substeps(d::DoubleWell, Δtlim) = d.substeps > 0 ? d.substeps : max(1, ceil(Int, d.T / (_cfl(d) * Δtlim) * (1 - 1e-12)))

# ---------------------------------------------------------------------------------------------------------------
# The step

# The optimiser of ReactionDiffusion (internal): the implicit step, the operator and the multiplier.
struct RDStep{T,D,N,V,LS,S,F,R} <: Optimizer
  τ::T                  # a number or a Parameter
  Δt::D                 # a number, a Parameter or :auto
  sensitivity::Symbol
  normalize::N
  volume::V             # ExactVolume or nothing
  mass::Symbol
  void_weight::Float64
  ls::LS
  solve::S              # nothing or (A, b, x, ctx, stale) -> x
  diffusion::F          # nothing or (φ, v) -> ∫(…)dΩ
  degree::Int
  gtol::Base.RefValue{Float64}   # the tolerance of the volume in units of the scaled constraint (set per problem)
  Δt_auto::Base.RefValue{Float64}  # Δt = :auto resolved for the problem (0.2 with an interface map, 0.4 without;
                                   # cfl Δt_lim with DoubleWell, set by init)
  reaction::R           # nothing or DoubleWell
end

# the step with the double-well reaction
const _DWStep = RDStep{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,DoubleWell}

_value(x::Real) = Float64(x)
_value(p::Parameter) = p[]

mutable struct RDState{NS,C,MT,MM,ME}
  m::Vector{Float64}            # lumped mass (free rows)
  K::MT                         # diffusion matrix (free × free)
  M::MM                         # consistent mass, or nothing
  A::MT                         # operator of the step, mass/Δt + τ ℓ² K
  Kmap::Vector{Int}             # positions of the entries of K, M and the diagonal in A.nzval
  Mmap::Vector{Int}
  Dmap::Vector{Int}
  kd::Vector{Float64}           # K_fd φ_d: the frozen values in the free rows
  area::Float64                 # |D|
  ℓ2::Float64                   # |D|^(2/d)
  unit::Bool                    # φ̌ = Δt (no Dirichlet values and the default diffusion)
  nfrozen::Int
  τ::Float64
  Δt::Float64                   # current (reduced after a rejected step)
  Δt_nom::Float64               # nominal
  ns::NS                        # numerical setup of ls (nothing with a solve of the user)
  stale::Bool                   # the matrix changed since the last call to the solve of the user
  φc::Vector{Float64}           # φ̌ = A⁻¹ m
  g::Vector{Float64}
  r::Vector{Float64}
  δ::Vector{Float64}
  φs::Vector{Float64}
  zt::Vector{Float64}
  zold::Vector{Float64}
  gv::Vector{Float64}
  λ::Float64
  shrunk::Bool
  kind::Symbol                  # :none, :linear, :newton or :illinois
  interface::Bool               # the objective reads the design through an interface map (phases)
  phase::Float64                # fraction of |D| whose phase changed in the last step (NaN without phases)
  meter::ME                     # the material read by the Volume, cell by cell (nothing if not available)
  zref::Vector{Float64}         # the design vector of the loop (the one given to init)
  dg::Float64                   # descent of the target per iteration (scaled units; Inf: none)
  ctx::C
  nsolves::Int
  nfactor::Int
  nvol::Int
  nmeter::Int                   # evaluations of the volume by the material meter (Step())
  vref::Float64                 # V_max + r of the native Volume (its value from the meter: (∫χ − vref)/vscale)
  vscale::Float64
  lb::Float64                   # the bounds of the design field (the grey measure, the double well)
  ub::Float64
  # the double well (empty or unused without it)
  gn::Vector{Float64}           # η C g/m
  substeps::Int                 # current substeps (1 once settled)
  nsub::Int                     # the substeps of an iteration before the settling (from T, or given)
  settled::Bool
  settle_iter::Int              # the iteration at which it settled (0: not yet)
  fwin::Vector{Float64}         # the last 10 objectives with the constraints met (oldest first)
  nwin::Int
  Δtlim::Float64                # the bound of Gershgorin of the explicit diffusion, 2/maxᵢ(τℓ² Σⱼ|Kᵢⱼ|/mᵢ)
  announced::Bool               # the event of Δt and of the width of the interface printed
end

# the union of the patterns of the matrices and the diagonal, with zero values
function _union_pattern(K, mats...)
  n = size(K, 1)
  colptr = Vector{Int}(undef, n + 1)
  rowval = Int[]
  colptr[1] = 1
  buf = Int[]
  for j in 1:n
    empty!(buf)
    push!(buf, j)
    for B in (K, mats...)
      for p in B.colptr[j]:(B.colptr[j+1]-1)
        push!(buf, B.rowval[p])
      end
    end
    sort!(buf)
    unique!(buf)
    append!(rowval, buf)
    colptr[j+1] = length(rowval) + 1
  end
  typeof(K)(n, n, colptr, rowval, zeros(length(rowval)))
end

# for each stored entry of B, its position in A.nzval (A has a larger pattern)
function _positions(A, B)
  pos = Vector{Int}(undef, length(B.nzval))
  for j in 1:size(B, 2)
    lo, hi = A.colptr[j], A.colptr[j+1] - 1
    for p in B.colptr[j]:(B.colptr[j+1]-1)
      q = searchsortedfirst(view(A.rowval, lo:hi), B.rowval[p]) + lo - 1
      pos[p] = q
    end
  end
  pos
end

_diagonal_positions(A) = [searchsortedfirst(view(A.rowval, A.colptr[j]:(A.colptr[j+1]-1)), j) + A.colptr[j] - 1
                          for j in 1:size(A, 2)]

_test_space(U) = U isa TrialFESpace ? U.space : U

function Optimizers.init(opt::RDStep, z, oracle)
  fo = _fo(oracle)
  fo isa FunctionalOracle ||
    throw(ArgumentError("ReactionDiffusion: the step works on the oracle of an Optimization (it assembles on the design space)"))
  zv = fo.z
  length(zv.blocks) == 1 || throw(ArgumentError("ReactionDiffusion: one design field (one block of z)"))
  U = zv.blocks[1].space
  V = _test_space(U)
  U0 = TrialFESpace(V)
  Ω = get_triangulation(U)
  d = num_cell_dims(Ω)
  dΩ = Measure(Ω, opt.degree)
  area = sum(∫(1.0)dΩ)
  m = collect(Float64, assemble_vector(v -> ∫(v)dΩ, V))
  dform = opt.diffusion === nothing ? ((u, v) -> ∫(∇(u) ⋅ ∇(v))dΩ) : opt.diffusion
  K = assemble_matrix(dform, U0, V)
  M = opt.mass === :consistent ? assemble_matrix((u, v) -> ∫(u * v)dΩ, U0, V) : nothing
  n = length(m)
  nd = num_dirichlet_dofs(U)
  kd = nd == 0 ? zeros(n) : collect(Float64, assemble_vector(v -> dform(FEFunction(U, zeros(num_free_dofs(U))), v), V))
  A = M === nothing ? _union_pattern(K) : _union_pattern(K, M)
  Kmap, Dmap = _positions(A, K), _diagonal_positions(A)
  Mmap = M === nothing ? Int[] : _positions(A, M)
  _resolve_Δt!(opt, (fo.J, (c.F for c in fo.cons)...))
  Δtlim, nsub = opt.reaction === nothing ? (NaN, 0) : _dw_resolve_Δt!(opt, K, m, area^(2 / d))
  τ, Δt = _check_τΔt(opt)
  ns = nothing
  if opt.solve === nothing
    _fill_operator!(A, K, M, m, Kmap, Mmap, Dmap, τ * area^(2 / d), Δt)
    ns = numerical_setup(symbolic_setup(opt.ls, A), A)
  end
  v() = zeros(n)
  dw = opt.reaction
  blk = zv.blocks[1]
  s = RDState(m, K, M, A, Kmap, Mmap, Dmap, kd, area, area^(2 / d), nd == 0 && opt.diffusion === nothing, nd, τ, Δt, Δt,
              ns, true, v(), v(), v(), v(), v(), v(), v(), v(), 0.0, false, :none,
              _has_interface_map(physical_fields(fo.J)), NaN, _material_meter(fo, z), z, Inf, fo.ctx, 0, 0, 0, 0, NaN, NaN,
              Float64(blk.lb), Float64(blk.ub), dw === nothing ? Float64[] : v(), nsub, nsub,
              false, 0, dw === nothing ? Float64[] : fill(NaN, _SETTLE_WINDOW), 0, Δtlim, false)
  if opt.solve === nothing
    s.nfactor += 1
    _unit_response!(opt, s)
  else
    _rebuild!(opt, s)
  end
  _rd_constraint!(opt, s, oracle, fo)
  s
end

# Δt = :auto: 0.2 when a functional reads the design through an interface map, 0.4 otherwise (decision of Jesús,
# 2026-11-24, from the study of the stopping iterations on the cantilever)
function _resolve_Δt!(opt::RDStep, Fs)
  opt.Δt === :auto || return opt
  opt.Δt_auto[] = any(F -> _has_interface_map(physical_fields(F)), Fs) ? 0.2 : 0.4
  opt
end
_resolve_Δt!(opt::_DWStep, Fs) = opt                 # from K and m, in init (_dw_resolve_Δt!)

# The bound of Gershgorin of the explicit diffusion, Δt_lim = 2/maxᵢ(τℓ² Σⱼ|Kᵢⱼ|/mᵢ) (K symmetric: the sums of the
# columns)
function _gershgorin_Δt(K, m, c)
  rmax = 0.0
  @inbounds for j in 1:size(K, 2)
    a = 0.0
    for p in K.colptr[j]:(K.colptr[j+1]-1)
      a += abs(K.nzval[p])
    end
    rmax = max(rmax, a / m[j])
  end
  c * rmax > 0 ? 2 / (c * rmax) : Inf
end

# Δt_lim, the substeps n of an iteration and Δt = :auto: T/n with n = ⌈T/(cfl Δt_lim)⌉, or cfl Δt_lim with the substeps
# given; with scheme = :explicit a Δt above Δt_lim is refused
function _dw_resolve_Δt!(opt::_DWStep, K, m, ℓ2)
  dw = opt.reaction
  Δtlim = _gershgorin_Δt(K, m, _value(opt.τ) * ℓ2)
  if opt.Δt === :auto
    isfinite(Δtlim) || throw(ArgumentError("ReactionDiffusion: with DoubleWell the substep comes from Δt_lim of the " *
                                           "diffusion, which needs τ > 0: give τ > 0 (or substeps = n and a number Δt)"))
  end
  n = isfinite(Δtlim) ? _dw_substeps(dw, Δtlim) : dw.substeps
  opt.Δt === :auto && (opt.Δt_auto[] = dw.substeps == 0 ? dw.T / n : _cfl(dw) * Δtlim)
  Δt = _Δt_value(opt)
  dw.scheme === :explicit && Δt > Δtlim * (1 + 1e-12) &&
    throw(ArgumentError("ReactionDiffusion: DoubleWell(scheme = :explicit) with Δt = $(_g(Δt)) above the bound of " *
                        "stability of the explicit diffusion, Δt_lim = $(_g(Δtlim)): use Δt ≤ Δt_lim (or Δt = :auto)"))
  Δtlim, n
end
_Δt_value(opt::RDStep) = opt.Δt === :auto ? opt.Δt_auto[] : _value(opt.Δt)

function _check_τΔt(opt)
  τ, Δt = _value(opt.τ), _Δt_value(opt)
  (isfinite(τ) && τ >= 0) || throw(ArgumentError("ReactionDiffusion: τ must be ≥ 0, got $τ"))
  (isfinite(Δt) && Δt > 0) || throw(ArgumentError("ReactionDiffusion: Δt must be positive, got $Δt"))
  τ, Δt
end

function _fill_operator!(A, K, M, m, Kmap, Mmap, Dmap, c, Δt)
  nz = A.nzval
  fill!(nz, 0.0)
  Kv = K.nzval
  @inbounds for k in eachindex(Kmap)
    nz[Kmap[k]] += c * Kv[k]
  end
  if M === nothing
    @inbounds for i in eachindex(Dmap)
      nz[Dmap[i]] += m[i] / Δt
    end
  else
    Mv = M.nzval
    @inbounds for k in eachindex(Mmap)
      nz[Mmap[k]] += Mv[k] / Δt
    end
  end
  A
end

# the operator for the current τ and Δt, factorised again, and φ̌
function _rebuild!(opt::RDStep, s::RDState)
  _fill_operator!(s.A, s.K, s.M, s.m, s.Kmap, s.Mmap, s.Dmap, s.τ * s.ℓ2, s.Δt)
  if opt.solve === nothing
    numerical_setup!(s.ns, s.A)
  else
    s.stale = true
  end
  s.nfactor += 1
  _unit_response!(opt, s)
  s
end

function _unit_response!(opt::RDStep, s::RDState)
  if s.unit
    fill!(s.φc, s.Δt)                           # A (Δt 1) = m exactly: no solve
  else
    _rd_solve!(s.φc, opt, s, s.m)
  end
  all(isfinite, s.φc) ||
    throw(OptimizationFailure("ReactionDiffusion: the solve of the step gave values that are not finite (the response φ̌ to a uniform source)"))
  minimum(s.φc) > 0 || throw(ArgumentError("ReactionDiffusion: the response of the step to a uniform source has " *
                                           "non-positive values (minimum $(minimum(s.φc))); use mass = :lumped"))
  s
end

function _rd_solve!(x, opt::RDStep, s::RDState, b)
  s.nsolves += 1
  if opt.solve === nothing
    solve!(x, s.ns, b)
  else
    y = opt.solve(s.A, b, x, s.ctx, s.stale)
    s.stale = false
    y === x || copyto!(x, y)
  end
  x
end

# the kind of root and the descent of the target, for the native constraint (if any)
function _rd_constraint!(opt::RDStep, s::RDState, oracle, fo)
  nc = num_constraints(oracle)
  nc == 0 && return s
  nc == 1 || throw(ArgumentError("ReactionDiffusion: the step treats one constraint; the oracle passes $nc"))
  info = constraint_info(oracle, 1)          # sets the scale
  handles(opt, info) ||
    throw(ArgumentError("ReactionDiffusion: the step treats one inequality g ≤ 0 that solves no state (the volume)"))
  i = oracle isa AugmentedLagrangianOracle ? oracle.native[1] : 1
  F = fo.cons[i].F
  θ = physical_fields(F)
  interface = _has_interface_map(θ)
  s.kind = linear_constraint(oracle, 1) !== nothing ? :linear :
           (θ isa PhysicalFields && !is_differentiable(θ)) ? :illinois : :newton
  # Step() with the Volume of the meter: the root evaluates the volume with the meter (exact, no allocation), so it
  # can bracket the jump of a single quadrature point
  me = s.meter
  if s.kind === :illinois && me !== nothing && me.kind === :step && me.fid == objectid(F)
    s.kind = :meter
    s.vref = F.J.vmax + _reference(fo.cons[i])
    s.vscale = get_scale(F)
  end
  ms = _max_step(opt, interface)
  s.dg = ms === nothing ? Inf : ms * s.area / get_scale(F)
  opt.gtol[] = opt.volume.tol * s.area / get_scale(F)
  s
end

# max_step = :auto: 0.02 with an interface map, 0.01 with the double well (decision of Jesús, 2026-11-25), none for a
# density
_max_step(opt::RDStep, interface) =
  opt.volume.max_step !== :auto ? opt.volume.max_step : opt.reaction !== nothing ? 0.01 : interface ? 0.02 : nothing

_has_interface_map(θ) = θ isa PhysicalFields && (has_ersatz(θ) || !is_differentiable(θ))
_has_step(θ) = θ isa PhysicalFields && !is_differentiable(θ)

# The material that the Volume constraint reads, cell by cell, from the nodal values of the design (no solve, no
# allocation): ∫_K χ dΩ with the quadrature of the measure of the Volume for Identity() (a density) and Step(), and
# |K| times the fraction of Ersatz(). The frozen zones are their nodal values (the image of the frozen value). Only
# when the Volume integrates over the cells of the design space, in their order, with one reference cell; the shape
# functions are those of the reference cell at its quadrature points (one matrix), the weights times |det J| one
# vector per cell (Identity() and Step(); Ersatz() keeps only the cell measures).
mutable struct _MaterialMeter{S}
  kind::Symbol                  # :identity, :step or :ersatz
  fid::UInt                     # objectid of the Volume measured
  stage::S                      # the ErsatzStage (nothing otherwise)
  data::Vector{Int32}           # cell dof ids of the design field (flat): positive free, negative Dirichlet
  ptrs::Vector{Int32}
  dv::Vector{Float64}           # the Dirichlet (frozen) values
  N::Matrix{Float64}            # shape functions of the reference cell at the quadrature points, q × a (empty: Ersatz)
  w::Matrix{Float64}            # quadrature weights times |det J|, q × cell (empty: Ersatz)
  W::Vector{Float64}            # cell measures
  cur::Vector{Float64}
  prev::Vector{Float64}
end

# the cells of Ωv are those of ΩU, in the same order
function _same_cells(Ωv, ΩU)
  Ωv === ΩU && return true
  num_cells(Ωv) == num_cells(ΩU) || return false
  try
    D = num_cell_dims(ΩU)
    collect(Gridap.Geometry.get_glue(Ωv, Val(D)).tface_to_mface) == collect(Gridap.Geometry.get_glue(ΩU, Val(D)).tface_to_mface) &&
      get_background_model(Ωv) === get_background_model(ΩU)
  catch
    false
  end
end

# the kind of meter of the Volume F (:identity, :step, :ersatz) and the Ersatz stage, or (:none, nothing)
function _meter_kind(F)
  θ = F.θ
  j = findfirst(==(F.J.field), getfield(θ, :names))
  chain = Tuple(st for st in getfield(θ, :chains)[j] if !(st isa MaskStage))
  core = Tuple(st for st in chain if !(st isa IdentityStage))
  isempty(core) && return :identity, nothing
  length(core) == 1 && core[1] isa StepStage && return :step, nothing
  length(core) == 1 && core[1] isa ErsatzStage && return :ersatz, core[1]
  :none, nothing
end

# the Volume of the meter among the functionals of a problem (exactly one, over the design domain), or nothing
function _meter_volume(Fs, U)
  vols = [F for F in Fs if is_volume(F)]
  length(vols) == 1 || return nothing
  F = vols[1]
  first(_meter_kind(F)) === :none && return nothing
  _same_cells(F.J.dΩ.quad.trian, get_triangulation(U)) || return nothing
  length(unique(Gridap.Geometry.get_cell_type(get_triangulation(U)))) == 1 || return nothing
  F
end

# the meter of the problem of `fo` (nothing when its Volume cannot be measured this way: see _meter_volume)
function _material_meter(fo, z)
  U = fo.z.blocks[1].space
  F = _meter_volume(Tuple(c.F for c in fo.cons), U)
  F === nothing && return nothing
  kind, stage = _meter_kind(F)
  ids = Gridap.Arrays.Table(get_cell_dof_ids(U))
  quad = F.J.dΩ.quad
  Ω = get_triangulation(U)
  nc = length(ids)
  x = Gridap.CellData.get_cell_points(quad)
  Jc = Gridap.Arrays.lazy_map(Gridap.Arrays.evaluate, Gridap.Arrays.lazy_map(∇, Gridap.Geometry.get_cell_map(Ω)),
                              Gridap.CellData.get_data(x))
  wq = quad.cell_weight
  W = [sum(wq[c] .* Gridap.TensorValues.meas.(Jc[c])) for c in 1:nc]
  if kind === :ersatz
    N, w = zeros(0, 0), zeros(0, 0)
  else
    N = Matrix{Float64}(get_fe_basis(_test_space(U))(x)[1])     # the same reference cell everywhere
    w = Matrix{Float64}(undef, size(N, 1), nc)
    for c in 1:nc
      w[:, c] .= wq[c] .* Gridap.TensorValues.meas.(Jc[c])
    end
  end
  me = _MaterialMeter(kind, objectid(F), stage, Vector{Int32}(ids.data), Vector{Int32}(ids.ptrs),
                      collect(Float64, get_dirichlet_dof_values(U)), N, w, W, zeros(nc), zeros(nc))
  _cell_material!(me.prev, me, z)                 # the material of the starting design
  me
end

@inline _nodal(data, z, dv, j) = (d = data[j]; d > 0 ? z[d] : dv[-d])

function _cell_material!(out, me::_MaterialMeter, z)
  data, ptrs, N, w = me.data, me.ptrs, me.N, me.w
  nq, na = size(N, 1), size(N, 2)
  @inbounds for c in eachindex(out)
    if me.kind === :ersatz
      st = me.stage
      out[c] = me.W[c] * _fraction(st, _cell_values(st, z, me.dv, c), c)[1]
      continue
    end
    p = ptrs[c]
    acc = 0.0
    for q in 1:nq
      φq = 0.0
      for a in 1:na
        φq += N[q, a] * _nodal(data, z, me.dv, p + a - 1)
      end
      acc += w[q, c] * (me.kind === :step ? (φq >= 0 ? 1.0 : 0.0) : φq)
    end
    out[c] = acc
  end
  out
end

# M_k = Σ_K |∫_K χ_k − ∫_K χ_{k−1}| / ∫ χ_k, and χ_k becomes the previous one
_material_change!(::Nothing, z) = NaN
function _material_change!(me::_MaterialMeter, z)
  _cell_material!(me.cur, me, z)
  num = sum(Broadcast.instantiate(Broadcast.broadcasted((a, b) -> abs(a - b), me.cur, me.prev)))
  den = sum(me.cur)
  copyto!(me.prev, me.cur)
  den > 0 ? num / den : NaN
end

# τ or Δt continued (Parameter), a rejected step, or the nominal Δt back after one
function _sync_parameters!(opt::RDStep, s::RDState, retry::Bool)
  τ, Δt = _check_τΔt(opt)
  changed = τ != s.τ || Δt != s.Δt_nom
  s.τ, s.Δt_nom = τ, Δt
  if changed || (!retry && s.Δt != Δt)
    retry || (s.Δt = Δt)
    _rebuild!(opt, s)
  end
  s
end

# the sensitivity of the step, with respect to the physical field (:physical) or to the design
function _rd_gradient!(g, opt::RDStep, s::RDState, oracle, z)
  ctx = s.ctx
  (opt.sensitivity === :physical && ctx isa OptContext) || return gradient!(g, oracle, z)
  # setfield! with constant names: no allocation (set_context! converts to the type of a field found at run time)
  setfield!(ctx, :void_weight, opt.void_weight)
  setfield!(ctx, :sensitivity, :physical)
  try
    gradient!(g, oracle, z)
  finally
    setfield!(ctx, :sensitivity, :design)
  end
end

# the normalisation C of the sensitivity: 0 only for a zero sensitivity (diffusion alone); not finite otherwise: a failure
function _normalization(opt::RDStep, s::RDState)
  C = opt.normalize(s.g, s.m, s.area)
  isfinite(C) && return Float64(C)
  all(iszero, s.g) && return 0.0
  throw(OptimizationFailure("ReactionDiffusion: the normalisation of the sensitivity (" *
                            (opt.normalize isa Function ? "function of the user" : string(opt.normalize)) * ") returned $C"))
end

# r = −τℓ²(K z + k_d) − C g, φ* = z + A⁻¹ r (the arithmetic of the step, without allocations, and one solve)
function _rd_rhs!(s::RDState, z, C)
  mul!(s.r, s.K, z)
  c = s.τ * s.ℓ2
  @. s.r = -c * (s.r + s.kd) - C * s.g
  s
end

function Optimizers.step!(opt::RDStep, s::RDState, z, oracle, ctx)
  retry = s.shrunk
  s.shrunk = false
  _sync_parameters!(opt, s, retry)
  f = objective(oracle, z)
  isfinite(f) || return StepInfo{1}(false, f, NaN, NaN, 0.0, s.Δt, 1, NaN, (0.0,), (NaN,))
  _rd_gradient!(s.g, opt, s, oracle, z)
  g0 = s.kind === :none ? 0.0 : constraint(oracle, 1, z)
  C = _normalization(opt, s)
  _rd_rhs!(s, z, C)
  _rd_solve!(s.δ, opt, s, s.r)
  @. s.φs = z + s.δ
  all(isfinite, s.φs) ||
    throw(OptimizationFailure("ReactionDiffusion: the solve of the step gave values that are not finite"))
  copyto!(s.zold, z)
  lb, ub = bounds(oracle)
  if s.kind === :none
    @. z = clamp(s.φs, lb, ub)
    λ, viol = NaN, 0.0
  else
    # the target, lowered at most dg; a step shrunk after a rejection lowers it in proportion to Δt
    gk = isfinite(s.dg) ? max(0.0, g0 - s.dg * s.Δt / s.Δt_nom) : 0.0
    λ, Gλ = _multiplier!(opt, s, oracle, gk, lb, ub)
    @. z = clamp(s.φs - λ * s.φc, lb, ub)
    s.λ = λ
    viol = max(0.0, Gλ + gk)
  end
  @. s.δ = z - s.zold
  # the fraction of the domain that changed phase (the lumped mass of the nodes where φ changed sign)
  s.phase = s.interface ?
    sum(Broadcast.instantiate(Broadcast.broadcasted((a, b, w) -> ((a >= 0) != (b >= 0)) ? w : 0.0, s.zold, z, s.m))) / s.area : NaN
  StepInfo{1}(true, f, NaN, NaN, maximum(abs, s.δ), s.Δt, 1, NaN, (viol,), (λ,))
end

# ---------------------------------------------------------------------------------------------------------------
# The step with the double well: `substeps` times diffusion (implicit with the factorised operator, or explicit) and
# then the semi-implicit pointwise reaction of Takezawa et al. (2010, Eqs. 44–45) with the multiplier λ of the volume
# inside the tilt term, found by a root on vectors at every substep

# The pointwise reaction of a value φ diffused to ψ, with the tilt G = η C g/m + λ, on φ̂ = (φ − lb)/(ub − lb):
# ∂φ̂/∂t = φ̂(1 − φ̂) r with r = 4W(φ̂ − ½) − 30 G φ̂(1 − φ̂), integrated semi-implicitly so that the result stays in
# [0, 1] for any Δt (the factor that would leave it is taken implicitly). Non-increasing in G; 0 and 1 are fixed points.
@inline function _dw_point(φ, ψ, G, Δt, W, lb, ub)
  w = ub - lb
  p = (φ - lb) / w
  a = clamp((ψ - lb) / w, 0.0, 1.0)
  r = 4W * (p - 0.5) - 30 * G * p * (1 - p)
  v = r <= 0 ? a / (1 - Δt * (1 - p) * r) : (a + Δt * p * r) / (1 + Δt * p * r)
  lb + w * clamp(v, 0.0, 1.0)
end

# the reaction of every free value for the multiplier λ into out (out may be z itself)
function _dw_react!(out, z, s::RDState, λ, W, lb, ub)
  φs, gn, Δt = s.φs, s.gn, s.Δt
  @inbounds for i in eachindex(out)
    out[i] = _dw_point(z[i], φs[i], gn[i] + λ, Δt, W, _bound(lb, i), _bound(ub, i))
  end
  out
end

# the diffusion of a substep, ψ into s.φs: implicit (m/Δt + τℓ²K) ψ = m z/Δt − τℓ² k_d (one back-substitution), or
# explicit ψ = z − Δt τℓ² (K z + k_d)/m
function _dw_diffuse!(opt::_DWStep, s::RDState, z)
  c = s.τ * s.ℓ2
  if opt.reaction.scheme === :explicit
    mul!(s.r, s.K, z)
    @. s.φs = z - s.Δt * c * (s.r + s.kd) / s.m
  else
    if s.M === nothing
      @. s.r = s.m * z / s.Δt - c * s.kd
    else
      mul!(s.r, s.M, z)
      @. s.r = s.r / s.Δt - c * s.kd
    end
    _rd_solve!(s.φs, opt, s, s.r)
  end
  all(isfinite, s.φs) ||
    throw(OptimizationFailure("ReactionDiffusion: the diffusion of a substep of the double well gave values that are not finite"))
  s
end

# G(λ) = g(R(z, ψ; η C g/m + λ)) − g_j: a linear constraint (vectors only) or one evaluated by the oracle
struct _DWLinearG{A,Z,L,U,S}
  a::A
  V::Float64
  z::Z
  s::S
  lb::L
  ub::U
  W::Float64
  gk::Float64
end
function (G::_DWLinearG)(λ)
  s = G.s
  _dw_react!(s.zt, G.z, s, λ, G.W, G.lb, G.ub)
  s.nvol += 1
  dot(G.a, s.zt) - G.V - G.gk
end

struct _DWOracleG{O,Z,L,U,S}
  oracle::O
  z::Z
  s::S
  lb::L
  ub::U
  W::Float64
  gk::Float64
end
function (G::_DWOracleG)(λ)
  s = G.s
  _dw_react!(s.zt, G.z, s, λ, G.W, G.lb, G.ub)
  s.nvol += 1
  constraint(G.oracle, 1, s.zt) - G.gk
end

# the multiplier of a substep: G is non-increasing in λ (the reaction is); the bracket ±Λ is wide enough to saturate
# the reaction at the interface. A target out of reach in this substep (it cannot remove or add that much material
# at the interface) takes the end of the bracket: the next substeps and iterations catch up (the violation is reported)
function _dw_multiplier!(opt::_DWStep, s::RDState, oracle, z, gj, lb, ub)
  W = opt.reaction.W
  sc = 1 + maximum(abs, s.gn)
  Λ = 1e3 * sc
  lo = gj > 0 ? -Λ : 0.0                       # descending target: equality; at the final one: λ ≥ 0
  maxit = max(opt.volume.maxiter, 100)
  λ, Gλ, status = if s.kind === :linear
    a, V = linear_constraint(oracle, 1)
    _root(_DWLinearG(a, V, z, s, lb, ub, W, gj), nothing, s.λ, lo, Λ, 0.1 * sc, opt.gtol[], maxit)
  else
    _root(_DWOracleG(oracle, z, s, lb, ub, W, gj), nothing, s.λ, lo, Λ, 0.1 * sc, opt.gtol[], maxit)
  end
  status === :nonfinite &&
    throw(OptimizationFailure("ReactionDiffusion: the constraint is not finite at a trial design of the multiplier"))
  λ, Gλ
end

function Optimizers.step!(opt::_DWStep, s::RDState, z, oracle, ctx)
  retry = s.shrunk
  s.shrunk = false
  _sync_parameters!(opt, s, retry)
  f = objective(oracle, z)
  isfinite(f) || return StepInfo{1}(false, f, NaN, NaN, 0.0, s.Δt, 1, NaN, (0.0,), (NaN,))
  _rd_gradient!(s.g, opt, s, oracle, z)
  g0 = s.kind === :none ? 0.0 : constraint(oracle, 1, z)
  C = _normalization(opt, s)
  dw = opt.reaction
  η = dw.η
  @. s.gn = η * C * s.g / s.m
  copyto!(s.zold, z)
  lb, ub = bounds(oracle)
  # the target of the iteration, lowered at most dg (in proportion to Δt after a rejection), reached linearly over
  # the substeps
  gk = (s.kind === :none || !isfinite(s.dg)) ? 0.0 : max(0.0, g0 - s.dg * s.Δt / s.Δt_nom)
  n = s.substeps
  λ = s.kind === :none ? NaN : s.λ
  viol = 0.0
  for j in 1:n
    _dw_diffuse!(opt, s, z)
    if s.kind === :none
      _dw_react!(z, z, s, 0.0, dw.W, lb, ub)
    else
      gj = g0 > gk ? gk + (g0 - gk) * (n - j) / n : gk
      λ, Gλ = _dw_multiplier!(opt, s, oracle, z, gj, lb, ub)
      s.λ = λ
      _dw_react!(z, z, s, λ, dw.W, lb, ub)
      viol = max(0.0, Gλ + gj)
    end
  end
  @. s.δ = z - s.zold
  StepInfo{1}(true, f, NaN, NaN, maximum(abs, s.δ), s.Δt, 1, NaN, (viol,), (λ,))
end

# grey measure of the design, M_nd = Σᵢ mᵢ 4φ̂ᵢ(1 − φ̂ᵢ)/|D| (nodal, the free values; frozen values are 0 or 1)
function grey_measure(::RDStep, s::RDState)
  lb, w = s.lb, s.ub - s.lb
  4 * sum(Broadcast.instantiate(Broadcast.broadcasted((z, m) -> (p = (z - lb) / w; m * p * (1 - p)), s.zref, s.m))) / s.area
end

# The settling of the double well and its events, after each accepted iteration: the first one says T, the substeps,
# Δt and the width of the interface in cells; then, with settle, one substep from the iteration where the constraints
# are met and the objective is steady over 10 iterations, or at 75 % of maxiter
const _SETTLE_WINDOW = 10
const _SETTLE_RANGE = 1e-3
const _SETTLE_FRACTION = 0.75

function iteration_events!(opt::_DWStep, s::RDState, k, maxiter, met, f)
  dw = opt.reaction
  ev = Tuple{String,Symbol}[]
  if !s.announced
    s.announced = true
    c = s.τ * s.ℓ2
    h = sqrt(s.Δtlim * c / 0.375)             # the mesh size of a uniform mesh of squares with this Δt_lim
    ξ = sqrt(c / (2dw.W))
    push!(ev, ("· double well: " * _dw_time_text(dw, s) * " (Δt_lim = $(_g(s.Δtlim)))" *
               ", width of the interface ξ = √(τℓ²/2W) = $(_g(ξ / h)) h", :cyan))
  end
  if dw.settle && !s.settled && s.substeps > 1
    if met && isfinite(f)
      _push_window!(s, f)
    else
      fill!(s.fwin, NaN)
      s.nwin = 0
    end
    steady = s.nwin >= _SETTLE_WINDOW && _window_range(s.fwin) <= _SETTLE_RANGE
    late = k >= ceil(Int, _SETTLE_FRACTION * maxiter)
    if steady || late
      s.substeps = 1
      s.settled = true
      s.settle_iter = k
      push!(ev, ("· double well settled at iteration $k (" * (steady ? "constraints met and objective steady over " *
                 "$(_SETTLE_WINDOW) iterations" : "$(round(Int, 100 * _SETTLE_FRACTION)) % of maxiter") *
                 "): one substep per iteration from now on", :cyan))
    end
  end
  isempty(ev) ? nothing : ev
end

# T = n × Δt per iteration (the nominal substeps)
_dw_time_text(dw::DoubleWell, s::RDState) =
  "T = $(_g(s.nsub * s.Δt_nom)) per iteration = $(s.nsub) substep" * (s.nsub == 1 ? "" : "s") * " × Δt = $(_g(s.Δt_nom))" *
  (dw.substeps == 0 ? "" : " (substeps given)")

# the default stop is held until the iteration after the settling (with settle and more than one substep): before it
# the frozen substeps keep a cycle of the interface and the windows of the stop can be met with the interface stuck
stop_allowed(opt::_DWStep, s::RDState, k) = !opt.reaction.settle || s.nsub == 1 || (s.settled && s.settle_iter < k)

function _push_window!(s::RDState, f)
  w = s.fwin
  @inbounds for i in 1:length(w)-1
    w[i] = w[i+1]
  end
  w[end] = f
  s.nwin = min(s.nwin + 1, length(w))
  s
end
function _window_range(w)
  lo, hi = extrema(w)
  (hi - lo) / maximum(abs, w)
end

# G(λ) = g(clamp(φ* − λ φ̌)) − g_k: a linear constraint (vectors only) or one evaluated by the oracle
struct _LinearG{A,L,U,S}
  a::A
  V::Float64
  s::S
  lb::L
  ub::U
  gk::Float64
end
function (G::_LinearG)(λ)
  s = G.s
  @. s.zt = clamp(s.φs - λ * s.φc, G.lb, G.ub)
  s.nvol += 1
  dot(G.a, s.zt) - G.V - G.gk
end

struct _OracleG{O,L,U,S}
  oracle::O
  s::S
  lb::L
  ub::U
  gk::Float64
end
function (G::_OracleG)(λ)
  s = G.s
  @. s.zt = clamp(s.φs - λ * s.φc, G.lb, G.ub)
  s.nvol += 1
  constraint(G.oracle, 1, s.zt) - G.gk
end

# G of Step() from the material meter: the volume of the clipped trial, quadrature point by quadrature point (the cell
# materials go to me.cur, which material_change recomputes)
struct _MeterG{M,L,U,S}
  me::M
  s::S
  lb::L
  ub::U
  gk::Float64
end
function (G::_MeterG)(λ)
  s = G.s
  @. s.zt = clamp(s.φs - λ * s.φc, G.lb, G.ub)
  s.nmeter += 1
  (sum(_cell_material!(G.me.cur, G.me, s.zt)) - s.vref) / s.vscale - G.gk
end

# the bracket [a, b] of Step() is closed when at most one quadrature point of the Volume changes side between the trials
# of a and b (the jump of G is then that of one point: b is the feasible end), or when the trials differ by less than
# 1e-12 (several points that change side at the same λ)
struct _QuadClosed{M,L,U,S}
  me::M
  s::S
  lb::L
  ub::U
end
function (c::_QuadClosed)(a, b)
  me, s = c.me, c.s
  (b - a) * maximum(s.φc) <= 1e-12 && return true
  crossings = 0
  data, ptrs, N = me.data, me.ptrs, me.N
  nq, na = size(N, 1), size(N, 2)
  @inbounds for k in 1:length(ptrs)-1               # ptrs has one entry more than the cells
    p = ptrs[k]
    for q in 1:nq
      φa = 0.0
      φb = 0.0
      for i in 1:na
        d = data[p+i-1]
        if d > 0
          φa += N[q, i] * clamp(s.φs[d] - a * s.φc[d], _bound(c.lb, d), _bound(c.ub, d))
          φb += N[q, i] * clamp(s.φs[d] - b * s.φc[d], _bound(c.lb, d), _bound(c.ub, d))
        else
          φa += N[q, i] * me.dv[-d]
          φb += N[q, i] * me.dv[-d]
        end
      end
      ((φa >= 0) != (φb >= 0)) && (crossings += 1) > 1 && return false
    end
  end
  true
end
_bound(b::Real, d) = b
_bound(b::AbstractVector, d) = b[d]

# −Σ aᵢ φ̌ᵢ over the values not clipped at the last trial
_clipped_slope(a, s::RDState, lb, ub) =
  -sum(Broadcast.instantiate(Broadcast.broadcasted((ai, c, t, l, u) -> (l < t < u) ? ai * c : 0.0, a, s.φc, s.zt, lb, ub)))

struct _LinearDG{A,L,U,S}
  a::A
  s::S
  lb::L
  ub::U
end
(D::_LinearDG)(λ) = _clipped_slope(D.a, D.s, D.lb, D.ub)

struct _OracleDG{O,L,U,S}
  oracle::O
  s::S
  lb::L
  ub::U
end
function (D::_OracleDG)(λ)
  constraint_gradient!(D.s.gv, D.oracle, 1, D.s.zt)      # at the last trial (the last call to constraint)
  _clipped_slope(D.s.gv, D.s, D.lb, D.ub)
end

function _multiplier!(opt::RDStep, s::RDState, oracle, gk, lb, ub)
  vol = opt.volume
  # bracket from the bounds: below λlo every value is at its upper bound, above λhi at its lower bound
  λlo = minimum(Broadcast.instantiate(Broadcast.broadcasted((p, u, c) -> (p - u) / c, s.φs, ub, s.φc)))
  λhi = maximum(Broadcast.instantiate(Broadcast.broadcasted((p, l, c) -> (p - l) / c, s.φs, lb, s.φc)))
  lo = gk > 0 ? λlo : 0.0                       # descending target: equality; at the final one: λ ≥ 0
  hi = max(λhi, lo)
  s0 = 0.02 / s.Δt                              # a shift of 0.02 of φ
  λ, Gλ, status = if s.kind === :linear
    a, V = linear_constraint(oracle, 1)
    _root(_LinearG(a, V, s, lb, ub, gk), _LinearDG(a, s, lb, ub), s.λ, lo, hi, s0, opt.gtol[], vol.maxiter, _Closed(s, lb, ub))
  elseif s.kind === :newton
    _root(_OracleG(oracle, s, lb, ub, gk), _OracleDG(oracle, s, lb, ub), s.λ, lo, hi, s0, opt.gtol[], vol.maxiter, _Closed(s, lb, ub))
  elseif s.kind === :meter
    # cheap and exact evaluations: as many as needed to bracket one quadrature point (at most 200)
    _root(_MeterG(s.meter, s, lb, ub, gk), nothing, s.λ, lo, hi, s0, opt.gtol[], max(vol.maxiter, 200),
          _QuadClosed(s.meter, s, lb, ub))
  else
    # Step() without the meter (a Volume over part of the domain, or another constraint): the width of the bracket
    _root(_OracleG(oracle, s, lb, ub, gk), nothing, s.λ, lo, hi, s0, opt.gtol[], vol.maxiter,
          (a, b) -> (b - a) * maximum(s.φc) <= 1e-9)
  end
  status === :nonfinite &&
    throw(OptimizationFailure("ReactionDiffusion: the constraint is not finite at a trial design of the multiplier"))
  status === :infeasible &&
    throw(ArgumentError("ReactionDiffusion: the constraint cannot be met within the bounds: with every free value at " *
                        "its lower bound it is still violated by $(@sprintf("%.3g", Gλ + gk)) (scaled)" *
                        _zone_hint(_fo(oracle))))
  λ, Gλ
end

# When the bracket [a, b] of the multiplier is closed. For a continuous G (linear, Ersatz()) a safeguard: the trial
# designs differ by less than 1e-12 of the range of the bounds. For Step(), see _QuadClosed.
struct _Closed{S}
  s::S
  range::Float64
end
_Closed(s, lb, ub) = _Closed(s, Float64(minimum(Broadcast.instantiate(Broadcast.broadcasted(-, ub, lb)))))
(c::_Closed)(a, b) = (b - a) * maximum(c.s.φc) <= 1e-12 * c.range

# _root(G, dG, x0, lo, hi, s0, tol, maxit, closed) -> (x, G(x), status)
# Root of a non-increasing `G` on `[lo, hi]` from `x0`: Newton with `dG` (if not `nothing`) while it stays inside the
# bracket, regula falsi (Illinois) otherwise; while only one side of the root is known, steps of `s0`, doubled each
# time, towards the other one (at most to `lo` or `hi`). Ends when `|G| ≤ tol`, after `maxit` evaluations, or when
# `closed(a, b)`; a tolerance that is not met ends on the feasible side (`G ≤ 0`; the bound `hi`, where every value is
# at its lower bound, if no feasible trial was seen). `status`: `:ok`; `:infeasible` when `G(hi) > tol`; `:upper`
# when `G(lo) < -tol` (then `x = lo`); `:nonfinite` when `G` is not finite.
function _root(G, dG, x0, lo, hi, s0, tol, maxit, closed = (a, b) -> b - a <= 1e-14 * max(1.0, abs(a), abs(b)))
  x = clamp(x0, lo, hi)
  gx = G(x)
  nev = 1
  isfinite(gx) || return x, gx, :nonfinite
  abs(gx) <= tol && return x, gx, :ok
  a, fa, b, fb = gx > 0 ? (x, gx, hi, NaN) : (lo, NaN, x, gx)
  st = s0
  side = 0
  while nev < maxit
    if gx > 0 && x >= hi
      return x, gx, :infeasible
    elseif gx < 0 && x <= lo
      return x, gx, (lo == 0 ? :ok : :upper)    # λ = 0: the inequality is inactive
    end
    newton = false
    xn = NaN
    if dG !== nothing
      d = dG(x)
      if d < 0
        xn = x - gx / d
        newton = a < xn < b
      end
    end
    if !newton
      if isnan(fb)
        xn = min(x + st, b)
        st *= 2
      elseif isnan(fa)
        xn = max(x - st, a)
        st *= 2
      else
        xn = (a * fb - b * fa) / (fb - fa)
        (a < xn < b) || (xn = (a + b) / 2)
      end
    end
    x = xn
    gx = G(x)
    nev += 1
    isfinite(gx) || return x, gx, :nonfinite
    abs(gx) <= tol && return x, gx, :ok
    if gx > 0
      a, fa = x, gx
      (!newton && side == 1 && !isnan(fb)) && (fb /= 2)     # Illinois: the end kept twice is halved
      side = 1
    else
      b, fb = x, gx
      (!newton && side == -1 && !isnan(fa)) && (fa /= 2)
      side = -1
    end
    (!isnan(fa) && !isnan(fb) && closed(a, b)) && break
  end
  gx > 0 && x >= hi && return x, gx, :infeasible
  if gx > tol                                  # end on the feasible side (b = hi if no feasible trial was seen)
    x = b
    gx = G(x)
    isfinite(gx) || return x, gx, :nonfinite
    gx > tol && return x, gx, :infeasible
  end
  x, gx, :ok
end

function Optimizers.shrink!(opt::RDStep, s::RDState, factor)
  s.Δt *= factor
  s.shrunk = true
  _rebuild!(opt, s)
  s
end

Optimizers.reset!(opt::RDStep, s::RDState) = (s.shrunk = false; s)
Optimizers.objective_changed!(::RDStep, s::RDState) = s
Optimizers.handles(opt::RDStep, c::ConstraintInfo) =
  opt.volume !== nothing && is_upper(c) && c.hi == 0 && c.design_only && !c.field
Optimizers.max_native(opt::RDStep) = opt.volume === nothing ? 0 : 1
Optimizers.constraint_tol(opt::RDStep) = opt.volume === nothing ? Inf : isnan(opt.gtol[]) ? opt.volume.tol : opt.gtol[]
phase_change(::RDStep, s::RDState) = s.phase
material_change(::RDStep, s::RDState) = _material_change!(s.meter, s.zref)
material_available(::RDStep, s::RDState) = s.meter !== nothing

# the native constraint of ExactVolume: the one named, or the one built with Volume (exactly one)
function _auto_native(opt::RDStep, problem, infos, names)
  opt.volume === nothing && return Int[]
  want = opt.volume.constraint
  if want !== nothing
    i = findfirst(==(want), names)
    i === nothing && throw(ArgumentError("ExactVolume: constraint = $(repr(want)) is not a constraint of the problem " *
                                         "(its constraints: $(join(names, ", ")))"))
    handles(opt, infos[i]) || throw(ArgumentError("ExactVolume: the constraint $want is not an inequality g ≤ 0 " *
                                                  "evaluated without a state"))
    return [i]
  end
  cand = [i for (i, c) in enumerate(values(problem.constraints)) if is_volume(c.F) && handles(opt, infos[i])]
  length(cand) == 1 && return cand
  isempty(cand) && throw(ArgumentError("ReactionDiffusion: ExactVolume (the default of volume) treats the constraint " *
    "built with Volume(…) (g ≤ 0), and the problem has none. To go on: add a Volume constraint; or name another " *
    "constraint g ≤ 0 with ExactVolume(; constraint = :name); or, for a problem without a volume (or without " *
    "constraints), use ReactionDiffusion(…; volume = nothing, lagrangian = AugmentedLagrangian(; native = ()))"))
  throw(ArgumentError("ReactionDiffusion: the problem has $(length(cand)) constraints built with Volume(…) " *
    "($(join(names[cand], ", "))): choose the one the step treats with ExactVolume(; constraint = :name)"))
end
Optimizers.computes_kkt(::RDStep) = false
_checks_linear_native(::RDStep) = true          # the step may leave a linear volume violated on purpose (max_step)
Optimizers.memory(::RDStep, s::RDState, z) = (λ = s.λ,)
function Optimizers.restore_memory!(::RDStep, s::RDState, z, v::NamedTuple)
  keys(v) == (:λ,) || throw(ArgumentError("ReactionDiffusion: the saved memory has entries $(keys(v)), expected (:λ,)"))
  s.λ = v.λ
  s
end
# with the double well: also the substeps, the settling and the window of the objective that decides it
const _DW_MEMORY_KEYS = (:λ, :substeps, :settled, :settle_iter, :nwin, ntuple(i -> Symbol("f", i), _SETTLE_WINDOW)...)
Optimizers.memory(::_DWStep, s::RDState, z) =
  NamedTuple{_DW_MEMORY_KEYS}((s.λ, Float64(s.substeps), Float64(s.settled), Float64(s.settle_iter), Float64(s.nwin),
                               ntuple(i -> s.fwin[i], _SETTLE_WINDOW)...))
function Optimizers.restore_memory!(::_DWStep, s::RDState, z, v::NamedTuple)
  keys(v) == _DW_MEMORY_KEYS ||
    throw(ArgumentError("ReactionDiffusion: the saved memory has entries $(keys(v)), expected $(_DW_MEMORY_KEYS) (DoubleWell)"))
  s.λ = v.λ
  s.substeps = Int(v.substeps)
  s.settled = v.settled != 0
  s.settle_iter = Int(v.settle_iter)
  s.nwin = Int(v.nwin)
  for i in 1:_SETTLE_WINDOW
    s.fwin[i] = v[Symbol("f", i)]
  end
  s
end

# ReactionDiffusion stops when the constraints are met, the material changes less than 2e-3 (relative) in each of the
# last 5 iterations and the objective varies less than 1e-3 (relative range) over the last 10, the windows counted with
# the constraints met (decision of Jesús, 2026-11-24, from the study of stopping criteria on the cantilever)
default_stop(::RDStep) = StopCriteria(material_tol = 2e-3, material_window = 5, objective_range = 1e-3, window = 10,
                                      combine = :all)
# with the double well the interface keeps a small cycle (1–2 % of the material per iteration before settling, a few
# per mille after it): material_tol = 1e-2, the rest the same (decision of Jesús, 2026-11-25)
default_stop(::_DWStep) = StopCriteria(material_tol = 1e-2, material_window = 5, objective_range = 1e-3, window = 10,
                                       combine = :all)

summary_note(::Optimizer, state) = ""
summary_note(::RDStep, s::RDState) =
  "  reaction–diffusion: $(s.nsolves) solves of the step, $(s.nfactor) factorisation" * (s.nfactor == 1 ? "" : "s") *
  ", $(s.nvol) evaluations of the constraint" * (s.nmeter == 0 ? "" : ", $(s.nmeter) of the volume by the material meter") *
  _dw_note(s)
_dw_note(s::RDState) = isempty(s.gn) ? "" :
  "; double well: $(s.nsub) substep" * (s.nsub == 1 ? "" : "s") * " of Δt = $(_g(s.Δt_nom)) per iteration (Δt_lim $(_g(s.Δtlim)))" *
  (s.settled ? ", 1 from iteration $(s.settle_iter + 1) (settled)" : "")

Base.show(io::IO, ::RDStep) = print(io, "RDStep")

# ---------------------------------------------------------------------------------------------------------------
# The method

const _RD_DEFAULTS = (Δt = :auto, normalize = L1Mean(), volume = ExactVolume(), mass = :lumped, void_weight = 1e-3,
                      ls = LUSolver(), solve = nothing, diffusion = nothing, degree = 2, lagrangian = AugmentedLagrangian(),
                      reaction = nothing)

const _RD_PRESETS = Dict{Symbol,NamedTuple}(
  :kim2020 => (Δt = 0.1, normalize = L2Norm(), mass = :consistent, volume = nothing,
               lagrangian = AugmentedLagrangian(Bertsekas(μ0 = 0.2, growth = 1.025, window = 1, progress = Inf, μmax = 2.0);
                                                native = ())),
  :takezawa2010 => (reaction = DoubleWell(W = 0.25, η = 20.0, scheme = :explicit, substeps = 20, settle = false),
                    normalize = _TakezawaNorm(), volume = ExactVolume(max_step = 0.01)))

"""
    ReactionDiffusion(; τ, sensitivity, Δt = :auto, normalize = L1Mean(), volume = ExactVolume(), mass = :lumped,
                      void_weight = 1e-3, ls = LUSolver(), solve = nothing, diffusion = nothing, degree = 2,
                      lagrangian = AugmentedLagrangian(), reaction = nothing)
    ReactionDiffusion(; τ, reaction = DoubleWell(), kwargs...)
    ReactionDiffusion(preset::Symbol; τ, sensitivity, kwargs...)

Reaction–diffusion method (Yamada et al. 2010, Choi et al. 2011, Otomori et al. 2015, Kim et al. 2020): the design
field `φ` (one nodal P1 field, `φ ∈ [lb, ub]`) evolves by an implicit step of

    ∂φ/∂t = −C g − λ + τ ℓ² Δφ,

that is `(mass/Δt + τ ℓ² K) δ = −τ ℓ² K z − C g` (`K` the diffusion matrix, `ℓ² = |D|^(2/d)` with `|D|` the measure of
the design domain, so `τ` is dimensionless), `φ* = z + δ`, and then `z = clamp(φ* − λ φ̌, lb, ub)` with the multiplier
`λ` of the volume found exactly at every step ([`ExactVolume`](@ref)). `g` is the derivative of the objective
(plus the terms of the augmented Lagrangian of the other constraints) and `C` its normalisation (`normalize`).

The material is read through the maps of the user: a level set (`bounds = (-1.0, 1.0)`, solid start `init = 1.0`) through
an interface map, `χ = :φ => Ersatz()` or `Step()`; or a density (`bounds = (0.0, 1.0)`) through
`Identity()` and a SIMP interpolation in the weak forms. Frozen zones of the design space (`FieldVariable(…; frozen)`)
are not variables: the step leaves them as they are.

Mandatory (modelling decisions):
- `τ`: diffusion coefficient (dimensionless; a number or a [`Parameter`](@ref HyperShape.Operators.Parameter), which a
  `Continuation` can change); larger values give simpler designs with thicker members.
- `sensitivity` (not with `reaction = DoubleWell(…)`, where it is `:design`): `:physical` takes the derivative with
  respect to the material, the interface map taken as the
  identity weighted by the material, `(w₀ + (1 − w₀) χ) ∂j/∂χ` (`w₀ = void_weight`): it nucleates holes, and the exact
  derivative through the map would refill the void (its value there is large: weak material, large strains); `:design`
  the exact derivative with respect to φ, for a density (without an interface map in the chain both are the same) and
  with the double well, where it is the default. `:design` with an interface map (`Ersatz()`, `Step()`) is not a
  method: the exact derivative lives on the cut cells and nothing holds the plateaus (`solve!` throws). The variant
  that only moves the boundary, from a start with holes ([`hole_seed`](@ref HyperShape.Variables.hole_seed)), is a
  density with `reaction = DoubleWell(…)` (then `sensitivity` is not mandatory). `hole_seed` also gives perforated
  starts for `:physical` (studies of the dependence on the start). [`check_consistency`](@ref HyperShape.Checks.check_consistency) always checks the
  exact derivative; with `Ersatz()` give it steps smaller than the default (`h0 = 1e-3`): larger ones cross the
  kinks of the cell fractions (a node changing sign) and the observed rates fall below 2 for a correct derivative.

Optional (each one replaceable, shown in the banner):
- `Δt`: pseudo-time step (number or `Parameter`); `:auto` (default) is `0.2` when the problem reads the design through
  an interface map and `0.4` otherwise (shown in the banner). With a smaller `Δt` the topology depends on the path: the
  design passes through metastable plateaus and the diffusion keeps merging holes long after it looks converged; halved by `RejectStep` when the forward problem does
  not converge at the new design, back to its value at the next step.
- `normalize`: [`L1Mean`](@ref)`(c)` (default `c = 1`), [`L2Norm`](@ref)`()`, [`MaxAbs`](@ref)`()`, or a function
  `(g, m, area) -> C` (`g` the derivative, `m` the lumped mass, `area` = `|D|`).
- `volume`: [`ExactVolume`](@ref)`(; max_step, tol, maxiter, constraint)`, or `nothing` (every constraint to the augmented
  Lagrangian; needs `lagrangian = AugmentedLagrangian(…; native = ())`).
- `mass`: `:lumped` (default; the response `φ̌` is positive and `G(λ)` monotone on non-obtuse meshes) or `:consistent`.
- `void_weight`: `w₀` of `sensitivity = :physical`, default `1e-3`.
- `ls`: the linear solver of the step (a Gridap `LinearSolver`), default `LUSolver()`: the operator is assembled once
  by HyperShape (Gridap's assembler, on the design space), factorised once and again only when `Δt` or `τ` change.
  `solve = (A, b, x, ctx, stale) -> x` replaces it (PETSc, an iterative solver of the user): `A` the operator, `b` the
  right-hand side, `x` the output, `stale` whether `A` changed since the last call.
- `diffusion = (φ, v) -> ∫(…)dΩ`: the bilinear form of the diffusion (default `∫(∇(φ) ⋅ ∇(v))dΩ` with a quadrature of
  degree `degree`), for an anisotropic diffusion or a Robin term.
- `lagrangian`: the [`AugmentedLagrangian`](@ref HyperShape.Optimizers.AugmentedLagrangian) of the constraints the
  step does not treat (everything but the volume).
- `reaction`: `nothing` (default: the linear reaction `−C g − λ` above) or [`DoubleWell`](@ref)`(…)`, the double-well
  reaction of Takezawa et al. (2010) for a density that only moves its interface, from a start with holes: then
  `sensitivity` defaults to `:design`, `normalize` to `L2Norm()`, `Δt = :auto` is `T/n` per substep (the time `T` of
  the well per iteration in `n` substeps) and the
  multiplier of `ExactVolume` acts inside the tilt term of the well (its `max_step = :auto` is `0.01`). See its
  docstring for the scheme, the substeps and the settling.

Presets (`ReactionDiffusion(:kim2020; τ, sensitivity)`): the keywords given replace those of the preset (the banner
says which). `:kim2020` (Kim et al. 2020, density): `Δt = 0.1, normalize = L2Norm(), mass = :consistent, volume = nothing,
lagrangian = AugmentedLagrangian(Bertsekas(μ0 = 0.2, growth = 1.025, window = 1, progress = Inf, μmax = 2.0); native = ())`;
`τ` is still mandatory (`5e-4` for their cantilever, `1e-3` for their half MBB beam). The script sets the rest:
a density `φ₀ = 0.5` with `bounds = (0.0, 1.0)`, SIMP `p = 3`, `Volume(…; scale = |Ω|)`.
`:takezawa2010` (Takezawa, Nishiwaki and Kitamura 2010, the double well): `reaction = DoubleWell(W = 0.25, η = 20,
scheme = :explicit, substeps = 20, settle = false)`, the normalisation `1/‖g‖_L²` (without `|D|`) and
`volume = ExactVolume(max_step = 0.01)`; the script sets `τ = κ/ℓ²` (their `κ = 1e-5` on the bridge 2 × 1 gives
`τ = 5e-6`), the start with holes, `k = max(φ³, 1e-5)` and 100 iterations (`stop = nothing`). Differences with the
paper: the multiplier of the volume is exact (instead of their rule for ζ) and the discretisation is P1 finite
elements with lumped mass (instead of finite differences).

Per iteration: one forward solve, one adjoint (none with `SelfAdjoint()`), one assembly of the sensitivity, one
product `K z` and one solve of the step (no factorisation); the volume costs a few evaluations of the constraint per
step (vectors only for a density; the map and an integral each with an interface map, plus one derivative per Newton
iteration with `Ersatz()`; the summary of `solve!` counts them). The default stop
([`default_stop`](@ref)) needs the constraints met, the change of the material `∫|χ_k − χ_{k−1}| / ∫χ_k` below `2e-3`
(`1e-2` with [`DoubleWell`](@ref)) in each of the last 5 iterations and the relative range of the objective below `1e-3` over the last 10, the windows
counted with the constraints met; `StopCriteria(rate_tol)`, `change_tol` and `phase_tol` are options. `√(τ|D|)` is the
length of the diffusion (a perimeter energy), not a minimum thickness: bars of one or two cells can remain. Neither the interface map nor the
stopping criterion is a recommendation: the choice will be validated with more cases. The starting design is not made feasible (the volume goes down from a solid start, `max_step` of `|D|` per
iteration). Checkpoints save the multiplier and the windows of the stop (a resumed run is the uninterrupted one).
`gtol` and the resolved `Δt` live in the method: one method object serves one `solve!` at a time (not concurrent
runs).
"""
struct ReactionDiffusion{S<:RDStep,L<:AugmentedLagrangian} <: AbstractMethod
  step::S
  lagrangian::L
  preset::Union{Nothing,Symbol}
  sources::Dict{Symbol,Symbol}     # :default, :preset or :given, per keyword with a default
end

ReactionDiffusion(; kwargs...) = _reaction_diffusion(nothing; kwargs...)
ReactionDiffusion(preset::Symbol; kwargs...) = _reaction_diffusion(preset; kwargs...)

function _reaction_diffusion(preset; kwargs...)
  allowed = (:τ, :sensitivity, keys(_RD_DEFAULTS)...)
  for k in keys(kwargs)
    k in allowed || throw(ArgumentError("ReactionDiffusion: unknown keyword $k; the keywords are $(join(allowed, ", "))"))
  end
  haskey(kwargs, :τ) || throw(ArgumentError("ReactionDiffusion: τ is mandatory (the diffusion, dimensionless; for example τ = 2e-4)"))
  p = if preset === nothing
    (;)
  else
    haskey(_RD_PRESETS, preset) ||
      throw(ArgumentError("ReactionDiffusion: unknown preset $(repr(preset)); the presets are " *
                          join(repr.(sort!(collect(keys(_RD_PRESETS)))), ", ")))
    _RD_PRESETS[preset]
  end
  sources = Dict{Symbol,Symbol}(k => (haskey(kwargs, k) ? :given : haskey(p, k) ? :preset : :default) for k in keys(_RD_DEFAULTS))
  val(k) = haskey(kwargs, k) ? kwargs[k] : haskey(p, k) ? p[k] : _RD_DEFAULTS[k]
  rx = val(:reaction)
  rx === nothing || rx isa DoubleWell ||
    throw(ArgumentError("ReactionDiffusion: reaction must be nothing or DoubleWell(…), got $(repr(rx))"))
  if rx === nothing
    haskey(kwargs, :sensitivity) ||
      throw(ArgumentError("ReactionDiffusion: sensitivity is mandatory: :physical (with respect to the material, nucleates " *
                          "holes) or :design (the exact derivative with respect to φ)"))
  else
    get(kwargs, :sensitivity, :design) === :physical &&
      throw(ArgumentError("ReactionDiffusion: sensitivity = :physical is not offered with DoubleWell: the double well " *
                          "already confines the reaction to the interface; leave sensitivity out (the exact derivative, :design)"))
  end
  τ, sens, Δt = kwargs[:τ], get(kwargs, :sensitivity, :design), val(:Δt)
  for (n, x, pos) in ((:τ, τ, false), (:Δt, Δt, true))
    (x isa Parameter || (n === :Δt && x === :auto)) && continue
    (x isa Real && isfinite(x) && (pos ? x > 0 : x >= 0)) ||
      throw(ArgumentError("ReactionDiffusion: $n must be a number $(pos ? "> 0" : "≥ 0") or a Parameter, got $(repr(x))"))
  end
  sens in (:physical, :design) || throw(ArgumentError("ReactionDiffusion: sensitivity must be :physical or :design, got $(repr(sens))"))
  rx isa DoubleWell && rx.substeps == 0 && Δt !== :auto &&
    throw(ArgumentError("ReactionDiffusion: with DoubleWell the substep is T/n (the time per iteration T of the well, " *
                        "n = ⌈T/(cfl Δt_lim)⌉): leave Δt out, or give DoubleWell(substeps = n) to set Δt per substep"))
  # with the double well the default normalisation is L2Norm() (decision of Jesús, 2026-11-25)
  norm = (rx !== nothing && sources[:normalize] === :default) ? L2Norm() : val(:normalize)
  (norm isa Union{L1Mean,L2Norm,MaxAbs,_TakezawaNorm} || norm isa Function) ||
    throw(ArgumentError("ReactionDiffusion: normalize must be L1Mean(c), L2Norm(), MaxAbs() or a function (g, m, area) -> C"))
  vol = val(:volume)
  vol === nothing || vol isa ExactVolume || throw(ArgumentError("ReactionDiffusion: volume must be ExactVolume(…) or nothing"))
  mass = val(:mass)
  mass in (:lumped, :consistent) || throw(ArgumentError("ReactionDiffusion: mass must be :lumped or :consistent, got $(repr(mass))"))
  w0 = val(:void_weight)
  (w0 isa Real && 0 <= w0 <= 1) || throw(ArgumentError("ReactionDiffusion: void_weight must be in [0, 1], got $(repr(w0))"))
  ls = val(:ls)
  ls isa LinearSolver || throw(ArgumentError("ReactionDiffusion: ls must be a Gridap LinearSolver (LUSolver(), …); for any other solver give solve = (A, b, x, ctx, stale) -> x"))
  solvef, diff = val(:solve), val(:diffusion)
  solvef === nothing || solvef isa Function || throw(ArgumentError("ReactionDiffusion: solve must be a function (A, b, x, ctx, stale) -> x"))
  diff === nothing || diff isa Function || throw(ArgumentError("ReactionDiffusion: diffusion must be a function (φ, v) -> ∫(…)dΩ"))
  deg = val(:degree)
  (deg isa Integer && deg >= 1) || throw(ArgumentError("ReactionDiffusion: degree must be an integer ≥ 1"))
  lag = val(:lagrangian)
  lag isa AugmentedLagrangian || throw(ArgumentError("ReactionDiffusion: lagrangian must be an AugmentedLagrangian"))
  vol === nothing && lag.native !== () &&
    throw(ArgumentError("ReactionDiffusion: volume = nothing gives every constraint to the augmented Lagrangian: " *
                        "say it with lagrangian = AugmentedLagrangian(…; native = ())"))
  step = RDStep(τ, Δt, sens, norm, vol, mass, Float64(w0), ls, solvef, diff, Int(deg), Ref(NaN), Ref(NaN), rx)
  ReactionDiffusion(step, lag, preset, sources)
end

stepper(m::ReactionDiffusion) = m.step
build_oracle(::ReactionDiffusion, problem) = FunctionalOracle(problem, LumpedMass())
augmented_lagrangian(m::ReactionDiffusion) = m.lagrangian
method_label(m::ReactionDiffusion) = m.preset === nothing ? "ReactionDiffusion" : "ReactionDiffusion($(repr(m.preset)))"
feasible_start(::ReactionDiffusion) = false
Base.show(io::IO, m::ReactionDiffusion) = print(io, method_label(m), "(τ = ", m.step.τ, ", sensitivity = ",
                                                repr(m.step.sensitivity), ")")

# a nodal P1 scalar field (H¹): one value per vertex of each cell, shared between cells
function _is_nodal_p1(U)
  Ω = get_triangulation(U)
  ncells = num_cells(Ω)
  ncells == 0 && return false
  nv = length(first(get_cell_coordinates(Ω)))
  all(c -> length(c) == nv, get_cell_dof_ids(U)) || return false
  num_free_dofs(U) + num_dirichlet_dofs(U) <= length(Gridap.Geometry.get_node_coordinates(Ω))
end

# the starting design changes sign (an interface for Ersatz(), whose derivative lives on the cut cells)
function _has_interface(z)
  U = z.blocks[1].space
  v, dv = get_values(z), get_dirichlet_dof_values(U)
  (any(<(0), v) || any(<(0), dv)) && (any(>=(0), v) || any(>=(0), dv))
end

function check_problem(m::ReactionDiffusion, problem)
  z = problem.z
  length(z.blocks) == 1 ||
    throw(ArgumentError("ReactionDiffusion: one design field (one block of z); the problem has $(length(z.blocks)): $(join(z.names, ", "))"))
  b = z.blocks[1]
  _is_nodal_p1(b.space) ||
    throw(ArgumentError("ReactionDiffusion: the design field must be a nodal P1 scalar field (ReferenceFE(lagrangian, " *
                        "Float64, 1), conformity = :H1)"))
  (isfinite(b.lb) && isfinite(b.ub)) || throw(ArgumentError("ReactionDiffusion: the bounds of z must be finite"))
  native = _native_split(problem, m.step, m.lagrangian)
  _set_tolerance!(m.step, problem, native)
  _resolve_Δt!(m.step, (problem.objective, (c.F for c in problem.constraints)...))
  m.step.reaction === nothing || return _check_double_well(m, problem)
  m.step.sensitivity === :design || return nothing
  for (j, (n, F)) in enumerate(pairs((objective = problem.objective, map(c -> c.F, problem.constraints)...)))
    j > 1 && (j - 1) in native && continue      # the native constraint is only evaluated (the root), never differentiated
    θ = physical_fields(F)
    who = n === :objective ? "the objective" : "the constraint $n"
    _has_step(θ) &&
      throw(ArgumentError("ReactionDiffusion: $who reads a field through Step(), which has no derivative with respect " *
                          "to the design: use sensitivity = :physical, or Ersatz()"))
    # decision of Jesús (2026-11-25): :design with an interface map and without the double well is not a method
    θ isa PhysicalFields && has_ersatz(θ) &&
      throw(ArgumentError("ReactionDiffusion: sensitivity = :design with an interface map is not a method of " *
                          "ReactionDiffusion: the exact derivative lives on the cut cells and nothing holds the plateaus. " *
                          "To move only the boundary from a start with holes use a density field (bounds = (0.0, 1.0), " *
                          "no interface map) with reaction = DoubleWell(…) and hole_seed; to nucleate holes use " *
                          "sensitivity = :physical"))
  end
  nothing
end

# DoubleWell: a density (no interface map in any functional) whose start has an interface (values on both sides of
# the middle of the bounds)
function _check_double_well(m::ReactionDiffusion, problem)
  for (n, F) in pairs((objective = problem.objective, map(c -> c.F, problem.constraints)...))
    who = n === :objective ? "the objective" : "the constraint $n"
    _has_interface_map(physical_fields(F)) &&
      throw(ArgumentError("ReactionDiffusion: $who reads the design through an interface map (Ersatz() or Step()): the " *
                          "double well with an interface map is not offered. Use a density field (bounds = (0.0, 1.0), " *
                          "Identity()) with an interpolation of the material in the weak forms (for example max(φ^3, 1e-5))"))
  end
  b = problem.z.blocks[1]
  mid = (b.lb + b.ub) / 2
  v, dv = get_values(problem.z), get_dirichlet_dof_values(b.space)
  ((any(<(mid), v) || any(<(mid), dv)) && (any(>=(mid), v) || any(>=(mid), dv))) ||
    throw(ArgumentError("ReactionDiffusion: the double well only moves the interface (its reaction vanishes at the bounds) " *
                        "and the starting design has none (all its values on one side of $(_g(mid))): start from a design " *
                        "with holes, hole_seed(Vφ; bounds = ($(b.lb), $(b.ub)), radius, spacing, …)"))
  nothing
end

# the tolerance of the volume in units of the scaled constraint, as soon as its scale is known (the banner shows it)
function _set_tolerance!(opt::RDStep, problem, native)
  opt.gtol[] = NaN
  (opt.volume === nothing || isempty(native)) && return opt
  s = get_scale(problem.constraints[native[1]].F)
  isfinite(s) || return opt                    # scale = :initial: set when the step starts
  Ω = get_triangulation(problem.z.blocks[1].space)
  opt.gtol[] = opt.volume.tol * sum(∫(1.0)Measure(Ω, 1)) / s
  opt
end

_rd_value(x::Real) = _g(x)
_rd_value(p::Parameter) = "Parameter(" * _g(p[]) * ")"

# the substeps from T, for the banner (before init: K and m assembled here once more)
function _dw_banner_time(opt::RDStep, dw::DoubleWell, U)
  V = _test_space(U)
  Ω = get_triangulation(U)
  d = num_cell_dims(Ω)
  dΩ = Measure(Ω, opt.degree)
  area = sum(∫(1.0)dΩ)
  m = collect(Float64, assemble_vector(v -> ∫(v)dΩ, V))
  dform = opt.diffusion === nothing ? ((u, v) -> ∫(∇(u) ⋅ ∇(v))dΩ) : opt.diffusion
  K = assemble_matrix(dform, TrialFESpace(V), V)
  Δtlim = _gershgorin_Δt(K, m, _value(opt.τ) * area^(2 / d))
  isfinite(Δtlim) || return ":auto (T = $(_g(dw.T)) per iteration; τ = 0)"
  n = _dw_substeps(dw, Δtlim)
  ":auto = T/n = $(_g(dw.T))/$n = $(_g(dw.T / n)) per substep (n = ⌈T/(cfl Δt_lim)⌉, cfl $(_g(_cfl(dw))), Δt_lim = $(_g(Δtlim)))"
end

function banner_lines(m::ReactionDiffusion, problem)
  opt = m.step
  src = m.sources
  mark(k) = src[k] === :default ? " (default)" : src[k] === :preset ? " ($(repr(m.preset)))" :
            (m.preset !== nothing && haskey(_RD_PRESETS[m.preset], k)) ? " (given, replaces $(repr(m.preset)))" : ""
  z = problem.z
  b = z.blocks[1]
  Ω = get_triangulation(b.space)
  area = sum(∫(1.0)Measure(Ω, 1))
  ℓ2 = area^(2 / num_cell_dims(Ω))
  lines = Tuple{String,Symbol}[]
  sens = opt.sensitivity === :physical ? ":physical (void weight $(_g(opt.void_weight))$(mark(:void_weight)))" : ":design"
  dw = opt.reaction
  Δtl = dw !== nothing && opt.Δt === :auto ? (dw.substeps == 0 ? _dw_banner_time(opt, dw, b.space) :
                                                ":auto = $(_g(_cfl(dw))) Δt_lim per substep") * mark(:Δt) :
        opt.Δt === :auto ? _g(opt.Δt_auto[]) * " (default: " * (opt.Δt_auto[] == 0.2 ? "interface map)" : "no interface map)") :
        _rd_value(opt.Δt) * mark(:Δt) * (dw === nothing ? "" : " per substep")
  dw === nothing || (sens = ":design (with DoubleWell)")
  push!(lines, ("method: sensitivity = $sens · Δt = $Δtl · τ = $(_rd_value(opt.τ)) " *
                "(× ℓ² = $(_g(ℓ2))) · clip [$(_g(b.lb)), $(_g(b.ub))]", :normal))
  if dw !== nothing
    push!(lines, ("reaction: $dw$(mark(:reaction)) · " * (dw.scheme === :implicit ? "implicit diffusion" : "explicit diffusion") *
                  " and semi-implicit reaction, multiplier inside the tilt · " *
                  (dw.settle ? "settles to 1 substep when the constraints are met and the objective range over " *
                               "$(_SETTLE_WINDOW) iterations is below $(_g(_SETTLE_RANGE)), or at $(round(Int, 100 * _SETTLE_FRACTION)) % of maxiter " *
                               "(the default stop waits for it)" :
                               "no settling"), :normal))
  end
  solver = opt.solve === nothing ? "$(nameof(typeof(opt.ls)))$(mark(:ls))" : "solve function of the user"
  nfrozen = b.zone === nothing ? 0 : b.zone.ndofs
  push!(lines, ("step operator: $(opt.mass) mass$(mark(:mass)) + " *
                (opt.diffusion === nothing ? "∇φ⋅∇v" : "diffusion of the user") * " · $solver" *
                (nfrozen == 0 ? "" : " · $nfrozen frozen values"), :normal))
  U = b.space
  if _meter_volume((problem.objective, (c.F for c in problem.constraints)...), U) === nothing
    push!(lines, ("material meter: none (no Volume over the design domain read through Identity(), Step() or Ersatz()): " *
                  "the default stop uses the constraints and the objective range", :normal))
  end
  push!(lines, ("normalize: " * (opt.normalize isa Function ? "function of the user" : string(opt.normalize)) * mark(:normalize), :normal))
  if opt.volume === nothing
    push!(lines, ("volume: not treated by the step (volume = nothing$(mark(:volume))): every constraint goes to the " *
                  "augmented Lagrangian", :normal))
  else
    native = _native_split(problem, opt, m.lagrangian)
    if isempty(native)
      push!(lines, ("volume: no constraint g ≤ 0 without a state to treat natively", :normal))
    else
      n = keys(problem.constraints)[native[1]]
      F = problem.constraints[native[1]].F
      θ = physical_fields(F)
      kind = is_linear(F) ? "Newton on the clipped vector" : _has_step(θ) ? "Illinois" : "Newton"
      ms = _max_step(opt, _has_interface_map(θ))
      opt.reaction === nothing || (kind = "inside the tilt of the double well, Illinois on " *
                                          (is_linear(F) ? "vectors" : "the constraint"))
      push!(lines, ("$n: native" * (opt.volume.constraint === nothing ? " (the Volume)" : " (named)") *
                    ", exact multiplier ($kind, tol $(_g_or_e(opt.volume.tol)) |D|), " *
                    (ms === nothing ? "target from the first iteration" : "target lowered $(_g(100 * ms)) % of |D| per iteration") *
                    mark(:volume), :normal))
    end
  end
  lines
end
