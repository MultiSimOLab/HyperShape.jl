# Reaction–diffusion method: an implicit step of a reaction–diffusion equation for a level set (or a density), with
# the volume met exactly at every step by a scalar multiplier (Choi et al. 2011, Yamada et al. 2010, Otomori et al.
# 2015, Kim et al. 2020, Cui et al. 2016 for the exact multiplier).

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

# ---------------------------------------------------------------------------------------------------------------
# Exact volume

"""
    ExactVolume(; max_step = :auto, tol = 1e-4, maxiter = 50)

How [`ReactionDiffusion`](@ref) treats the volume (its default): natively, with a scalar multiplier `λ` found at
every step so that the clipped design `z = clamp(φ* − λ φ̌, lb, ub)` meets the constraint (`φ*` the design after the
reaction–diffusion step, `φ̌ = A⁻¹ m` the response of the step to a uniform unit source). It takes the first
constraint of the problem that is an inequality `g ≤ 0` evaluated without a state (the volume), linear or not; the
others go to the augmented Lagrangian of the method.

- `max_step`: how much the target may go down per iteration, as a fraction of the measure `|D|` of the design
  domain: the target of iteration `k` is `max(V_obj, V_{k−1} − max_step |D|)`, met with equality while it goes down;
  at `V_obj` the constraint is `g ≤ 0` with `λ ≥ 0`. `:auto` (default): `0.01` when the constraint reads the
  material through an interface map (`Step()`, `Ersatz()`), `nothing` otherwise (a density: the target from the
  first iteration). `nothing`: the target from the first iteration. A number in `(0, 1]`.
- `tol`: tolerance of the root, in the units of the scaled constraint (the loop converges only with the
  constraint met to it).
- `maxiter`: evaluations of the constraint per step at most.

The root of `G(λ) = g(clamp(φ* − λ φ̌)) − g_k` (non-increasing in λ) starts from the previous `λ`, inside the bracket
given by the bounds (all values at the upper or at the lower bound): a linear constraint (a density) by Newton on the
clipped vector (only vectors, no solve; the algorithm of Cui et al. 2016); a nonlinear one with a derivative
(`Ersatz()`) by Newton with the derivative of the constraint, guarded by the bracket; `Step()` by regula falsi
(Illinois), which ends on the feasible side when the tolerance cannot be met (a piecewise-constant volume). Each
evaluation of a nonlinear constraint updates the maps and integrates (no state is solved).
"""
struct ExactVolume
  max_step::Union{Symbol,Nothing,Float64}
  tol::Float64
  maxiter::Int
  function ExactVolume(; max_step = :auto, tol::Real = 1e-4, maxiter::Integer = 50)
    (max_step === :auto || max_step === nothing || (max_step isa Real && 0 < max_step <= 1)) ||
      throw(ArgumentError("ExactVolume: max_step must be :auto, nothing or a fraction of |D| in (0, 1], got $(repr(max_step))"))
    (isfinite(tol) && tol > 0) || throw(ArgumentError("ExactVolume: tol must be positive, got $tol"))
    maxiter >= 2 || throw(ArgumentError("ExactVolume: maxiter must be at least 2, got $maxiter"))
    new(max_step isa Real ? Float64(max_step) : max_step, Float64(tol), Int(maxiter))
  end
end

Base.show(io::IO, v::ExactVolume) =
  print(io, "ExactVolume(max_step = ", repr(v.max_step), ", tol = ", v.tol, ", maxiter = ", v.maxiter, ")")

# ---------------------------------------------------------------------------------------------------------------
# The step

# The optimiser of ReactionDiffusion (internal): the implicit step, the operator and the multiplier.
struct RDStep{T,D,N,V,LS,S,F} <: Optimizer
  τ::T                  # a number or a Parameter
  Δt::D                 # a number or a Parameter
  sensitivity::Symbol
  normalize::N
  volume::V             # ExactVolume or nothing
  mass::Symbol
  void_weight::Float64
  ls::LS
  solve::S              # nothing or (A, b, x, ctx, stale) -> x
  diffusion::F          # nothing or (φ, v) -> ∫(…)dΩ
  degree::Int
end

_value(x::Real) = Float64(x)
_value(p::Parameter) = p[]

mutable struct RDState{NS,C}
  m::Vector{Float64}            # lumped mass (free rows)
  K::Any                        # diffusion matrix (free × free)
  M::Any                        # consistent mass, or nothing
  A::Any                        # operator of the step, mass/Δt + τ ℓ² K
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
  dg::Float64                   # descent of the target per iteration (scaled units; Inf: none)
  ctx::C
  nsolves::Int
  nfactor::Int
  nvol::Int
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
  τ, Δt = _check_τΔt(opt)
  ns = nothing
  if opt.solve === nothing
    _fill_operator!(A, K, M, m, Kmap, Mmap, Dmap, τ * area^(2 / d), Δt)
    ns = numerical_setup(symbolic_setup(opt.ls, A), A)
  end
  v() = zeros(n)
  s = RDState(m, K, M, A, Kmap, Mmap, Dmap, kd, area, area^(2 / d), nd == 0 && opt.diffusion === nothing, nd, τ, Δt, Δt,
              ns, true, v(), v(), v(), v(), v(), v(), v(), v(), 0.0, false, :none, Inf, fo.ctx, 0, 0, 0)
  if opt.solve === nothing
    s.nfactor += 1
    _unit_response!(opt, s)
  else
    _rebuild!(opt, s)
  end
  _rd_constraint!(opt, s, oracle, fo)
  s
end

function _check_τΔt(opt)
  τ, Δt = _value(opt.τ), _value(opt.Δt)
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
  ms = opt.volume.max_step === :auto ? (interface ? 0.01 : nothing) : opt.volume.max_step
  s.dg = ms === nothing ? Inf : ms * s.area / get_scale(F)
  s
end

_has_interface_map(θ) = θ isa PhysicalFields && (has_ersatz(θ) || !is_differentiable(θ))
_has_step(θ) = θ isa PhysicalFields && !is_differentiable(θ)

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
  set_context!(ctx, :void_weight, opt.void_weight)
  set_context!(ctx, :sensitivity, :physical)
  try
    gradient!(g, oracle, z)
  finally
    set_context!(ctx, :sensitivity, :design)
  end
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
  C = opt.normalize(s.g, s.m, s.area)
  isfinite(C) || (C = 0.0)                     # a zero sensitivity: diffusion only
  _rd_rhs!(s, z, C)
  _rd_solve!(s.δ, opt, s, s.r)
  @. s.φs = z + s.δ
  copyto!(s.zold, z)
  lb, ub = bounds(oracle)
  if s.kind === :none
    @. z = clamp(s.φs, lb, ub)
    λ, viol = NaN, 0.0
  else
    gk = isfinite(s.dg) ? max(0.0, g0 - s.dg) : 0.0
    λ, Gλ = _multiplier!(opt, s, oracle, gk, lb, ub)
    @. z = clamp(s.φs - λ * s.φc, lb, ub)
    s.λ = λ
    viol = max(0.0, Gλ + gk)
  end
  @. s.δ = z - s.zold
  StepInfo{1}(true, f, NaN, NaN, maximum(abs, s.δ), s.Δt, 1, NaN, (viol,), (λ,))
end

# G(λ) = g(clamp(φ* − λ φ̌)) − g_k: a linear constraint (vectors only) or one evaluated by the oracle
struct _LinearG{A,L,U}
  a::A
  V::Float64
  s::RDState
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

struct _OracleG{O,L,U}
  oracle::O
  s::RDState
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

# −Σ aᵢ φ̌ᵢ over the values not clipped at the last trial
_clipped_slope(a, s::RDState, lb, ub) =
  -sum(Broadcast.instantiate(Broadcast.broadcasted((ai, c, t, l, u) -> (l < t < u) ? ai * c : 0.0, a, s.φc, s.zt, lb, ub)))

struct _LinearDG{A,L,U}
  a::A
  s::RDState
  lb::L
  ub::U
end
(D::_LinearDG)(λ) = _clipped_slope(D.a, D.s, D.lb, D.ub)

struct _OracleDG{O,L,U}
  oracle::O
  s::RDState
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
    _root(_LinearG(a, V, s, lb, ub, gk), _LinearDG(a, s, lb, ub), s.λ, lo, hi, s0, vol.tol, vol.maxiter, maximum(s.φc))
  elseif s.kind === :newton
    _root(_OracleG(oracle, s, lb, ub, gk), _OracleDG(oracle, s, lb, ub), s.λ, lo, hi, s0, vol.tol, vol.maxiter, maximum(s.φc))
  else
    _root(_OracleG(oracle, s, lb, ub, gk), nothing, s.λ, lo, hi, s0, vol.tol, vol.maxiter, maximum(s.φc))
  end
  status === :infeasible &&
    throw(ArgumentError("ReactionDiffusion: the constraint cannot be met within the bounds: with every free value at " *
                        "its lower bound it is still violated by $(@sprintf("%.3g", Gλ + gk)) (scaled)" *
                        _zone_hint(_fo(oracle))))
  λ, Gλ
end

# _root(G, dG, x0, lo, hi, s0, tol, maxit, xscale = 1) -> (x, G(x), status)
# Root of a non-increasing `G` on `[lo, hi]` from `x0`: Newton with `dG` (if not `nothing`) while it stays inside the
# bracket, regula falsi (Illinois) otherwise; while only one side of the root is known, steps of `s0`, doubled each
# time, towards the other one (at most to `lo` or `hi`). Ends when `|G| ≤ tol`, after `maxit` evaluations, or when the
# bracket closes (`(b − a) xscale ≤ 1e-9`: the trial designs differ by less than that); a tolerance that cannot be met
# (a piecewise-constant `G`) ends on the feasible side (`G ≤ 0`). `status`: `:ok`; `:infeasible` when `G(hi) > tol`;
# `:upper` when `G(lo) < -tol` (then `x = lo`).
function _root(G, dG, x0, lo, hi, s0, tol, maxit, xscale = 1.0)
  x = clamp(x0, lo, hi)
  gx = G(x)
  nev = 1
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
    (!isnan(fa) && !isnan(fb) && (b - a) * xscale <= 1e-9) && break     # the trials differ by less than 1e-9
  end
  gx > 0 && x >= hi && return x, gx, :infeasible
  if gx > tol && !isnan(fb)                    # end on the feasible side
    x = b
    gx = G(x)
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
Optimizers.constraint_tol(opt::RDStep) = opt.volume === nothing ? Inf : opt.volume.tol
Optimizers.computes_kkt(::RDStep) = false
Optimizers.memory(::RDStep, s::RDState, z) = (λ = s.λ,)
function Optimizers.restore_memory!(::RDStep, s::RDState, z, v::NamedTuple)
  keys(v) == (:λ,) || throw(ArgumentError("ReactionDiffusion: the saved memory has entries $(keys(v)), expected (:λ,)"))
  s.λ = v.λ
  s
end

"""
    default_stop(::RDStep)

[`ReactionDiffusion`](@ref) stops when `maximum(abs, Δz) < 1e-2` and the constraints hold to their tolerances
(`StopCriteria(change_tol = 1e-2)`). A criterion on the objective stops too early: the objective settles before the
topology does.
"""
default_stop(::RDStep) = StopCriteria(change_tol = 1e-2)

summary_note(::Optimizer, state) = ""
summary_note(::RDStep, s::RDState) =
  "  reaction–diffusion: $(s.nsolves) solves of the step, $(s.nfactor) factorisation" * (s.nfactor == 1 ? "" : "s") *
  ", $(s.nvol) evaluations of the constraint"

Base.show(io::IO, ::RDStep) = print(io, "RDStep")

# ---------------------------------------------------------------------------------------------------------------
# The method

const _RD_DEFAULTS = (Δt = 0.1, normalize = L1Mean(), volume = ExactVolume(), mass = :lumped, void_weight = 1e-3,
                      ls = LUSolver(), solve = nothing, diffusion = nothing, degree = 2, lagrangian = AugmentedLagrangian())

const _RD_PRESETS = Dict{Symbol,NamedTuple}(
  :kim2020 => (Δt = 0.1, normalize = L2Norm(), mass = :consistent, volume = nothing,
               lagrangian = AugmentedLagrangian(Bertsekas(μ0 = 0.2, growth = 1.025, window = 1, progress = Inf, μmax = 2.0);
                                                native = ())))

"""
    ReactionDiffusion(; τ, sensitivity, Δt = 0.1, normalize = L1Mean(), volume = ExactVolume(), mass = :lumped,
                      void_weight = 1e-3, ls = LUSolver(), solve = nothing, diffusion = nothing, degree = 2,
                      lagrangian = AugmentedLagrangian())
    ReactionDiffusion(preset::Symbol; τ, sensitivity, kwargs...)

Reaction–diffusion method (Yamada et al. 2010, Choi et al. 2011, Otomori et al. 2015, Kim et al. 2020): the design
field `φ` (one nodal P1 field, `φ ∈ [lb, ub]`) evolves by an implicit step of

    ∂φ/∂t = −C g − λ + τ ℓ² Δφ,

that is `(mass/Δt + τ ℓ² K) δ = −τ ℓ² K z − C g` (`K` the diffusion matrix, `ℓ² = |D|^(2/d)` with `|D|` the measure of
the design domain, so `τ` is dimensionless), `φ* = z + δ`, and then `z = clamp(φ* − λ φ̌, lb, ub)` with the multiplier
`λ` of the volume found exactly at every step ([`ExactVolume`](@ref)). `g` is the derivative of the objective
(plus the terms of the augmented Lagrangian of the other constraints) and `C` its normalisation (`normalize`).

The material is read through the maps of the user: a level set (`bounds = (-1.0, 1.0)`, solid start `init = 1.0`) through
an interface map, `χ = :φ => Ersatz()` (recommended) or `Step()`; or a density (`bounds = (0.0, 1.0)`) through
`Identity()` and a SIMP interpolation in the weak forms. Frozen zones of the design space (`FieldVariable(…; frozen)`)
are not variables: the step leaves them as they are.

Mandatory (modelling decisions):
- `τ`: diffusion coefficient (dimensionless; a number or a [`Parameter`](@ref HyperShape.Operators.Parameter), which a
  `Continuation` can change); larger values give simpler designs with thicker members.
- `sensitivity`: `:physical` takes the derivative with respect to the material, the interface map taken as the
  identity weighted by the material, `(w₀ + (1 − w₀) χ) ∂j/∂χ` (`w₀ = void_weight`): it nucleates holes, and the exact
  derivative through the map would refill the void (its value there is large: weak material, large strains); `:design`
  the exact derivative with respect to φ (with `Ersatz()`, nonzero only on the cells cut by the interface: no holes are
  nucleated, so the starting design must have an interface; `Step()` has none). Without an interface map in the chain
  (a density) both are the same.

Optional (each one replaceable, shown in the banner):
- `Δt`: pseudo-time step (number or `Parameter`), default `0.1`; halved by `RejectStep` when the forward problem does
  not converge at the new design, back to its value at the next step.
- `normalize`: [`L1Mean`](@ref)`(c)` (default `c = 1`), [`L2Norm`](@ref)`()`, [`MaxAbs`](@ref)`()`, or a function
  `(g, m, area) -> C` (`g` the derivative, `m` the lumped mass, `area` = `|D|`).
- `volume`: [`ExactVolume`](@ref)`(; max_step, tol, maxiter)`, or `nothing` (every constraint to the augmented
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

Presets (`ReactionDiffusion(:kim2020; τ, sensitivity)`): the keywords given replace those of the preset (the banner
says which). `:kim2020` (Kim et al. 2020, density): `Δt = 0.1, normalize = L2Norm(), mass = :consistent, volume = nothing,
lagrangian = AugmentedLagrangian(Bertsekas(μ0 = 0.2, growth = 1.025, window = 1, progress = Inf, μmax = 2.0); native = ())`;
`τ` is still mandatory (`5e-4` for their cantilever, `1e-3` for their half MBB beam). The script sets the rest:
a density `φ₀ = 0.5` with `bounds = (0.0, 1.0)`, SIMP `p = 3`, `Volume(…; scale = |Ω|)`.

Per iteration: one forward solve, one adjoint (none with `SelfAdjoint()`), one assembly of the sensitivity, one
product `K z` and one solve of the step (no factorisation); the volume costs 2–5 evaluations (vectors only for a
density; the map and an integral each with an interface map, plus one derivative per Newton iteration with
`Ersatz()`). The default stop is `StopCriteria(change_tol = 1e-2)` with the constraints met ([`default_stop`](@ref)); with
`Step()` it is rarely met (the interface jumps between nodes) and runs usually end at `maxiter`: `Ersatz()` is
recommended. The starting design is not made feasible (the volume goes down from a solid start, `max_step` of `|D|` per
iteration). Checkpoints save the multiplier (a resumed run is the uninterrupted one).
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
  haskey(kwargs, :sensitivity) ||
    throw(ArgumentError("ReactionDiffusion: sensitivity is mandatory: :physical (with respect to the material, nucleates " *
                        "holes) or :design (the exact derivative with respect to φ)"))
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
  τ, sens, Δt = kwargs[:τ], kwargs[:sensitivity], val(:Δt)
  for (n, x, pos) in ((:τ, τ, false), (:Δt, Δt, true))
    x isa Parameter && continue
    (x isa Real && isfinite(x) && (pos ? x > 0 : x >= 0)) ||
      throw(ArgumentError("ReactionDiffusion: $n must be a number $(pos ? "> 0" : "≥ 0") or a Parameter, got $(repr(x))"))
  end
  sens in (:physical, :design) || throw(ArgumentError("ReactionDiffusion: sensitivity must be :physical or :design, got $(repr(sens))"))
  norm = val(:normalize)
  (norm isa Union{L1Mean,L2Norm,MaxAbs} || norm isa Function) ||
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
  step = RDStep(τ, Δt, sens, norm, vol, mass, Float64(w0), ls, solvef, diff, Int(deg))
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
  cs = values(problem.constraints)
  m.step.sensitivity === :design || return nothing
  for (j, (n, F)) in enumerate(pairs((objective = problem.objective, map(c -> c.F, problem.constraints)...)))
    j > 1 && (j - 1) in native && continue      # the native constraint is only evaluated (the root), never differentiated
    θ = physical_fields(F)
    who = n === :objective ? "the objective" : "the constraint $n"
    _has_step(θ) &&
      throw(ArgumentError("ReactionDiffusion: $who reads a field through Step(), which has no derivative with respect " *
                          "to the design: use sensitivity = :physical, or Ersatz()"))
    θ isa PhysicalFields && has_ersatz(θ) && !_has_interface(z) &&
      throw(ArgumentError("ReactionDiffusion: with sensitivity = :design the derivative through Ersatz() lives on the " *
                          "cells cut by the interface, and the starting design has none (φ does not change sign): no " *
                          "hole can appear. Use sensitivity = :physical, or start from a design with holes"))
  end
  nothing
end

_rd_value(x::Real) = _g(x)
_rd_value(p::Parameter) = "Parameter(" * _g(p[]) * ")"

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
  push!(lines, ("method: sensitivity = $sens · Δt = $(_rd_value(opt.Δt))$(mark(:Δt)) · τ = $(_rd_value(opt.τ)) " *
                "(× ℓ² = $(_g(ℓ2))) · clip [$(_g(b.lb)), $(_g(b.ub))]", :normal))
  solver = opt.solve === nothing ? "$(nameof(typeof(opt.ls)))$(mark(:ls))" : "solve function of the user"
  nfrozen = b.zone === nothing ? 0 : b.zone.ndofs
  push!(lines, ("step operator: $(opt.mass) mass$(mark(:mass)) + " *
                (opt.diffusion === nothing ? "∇φ⋅∇v" : "diffusion of the user") * " · $solver" *
                (nfrozen == 0 ? "" : " · $nfrozen frozen values"), :normal))
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
      ms = opt.volume.max_step === :auto ? (_has_interface_map(θ) ? 0.01 : nothing) : opt.volume.max_step
      push!(lines, ("$n: native, exact multiplier ($kind, tol $(_g_or_e(opt.volume.tol))), " *
                    (ms === nothing ? "target from the first iteration" : "target lowered $(_g(100 * ms)) % of |D| per iteration") *
                    mark(:volume), :normal))
    end
  end
  F = problem.objective
  any(G -> _has_step(physical_fields(G)), (F, (c.F for c in problem.constraints)...)) &&
    push!(lines, ("with Step() the change criterion is rarely met; runs usually end at maxiter; Ersatz() is recommended", :yellow))
  lines
end
