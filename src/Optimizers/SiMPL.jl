"""
    SiMPL(; linesearch = :armijo, c1 = 1e-4, max_backtracks = 30, root_tol = 1e-12, max_root_iterations = 100)

Sigmoidal mirror descent with a projected latent variable (Keith, Kim, Lazarov and Surowiec, arXiv:2409.19341;
Kim, Lazarov, Surowiec and Keith, arXiv:2411.19421).

The design is written as `z = lb + (ub - lb) σ(ψ)` with the sigmoid `σ(x) = 1/(1 + e⁻ˣ)`, so every iterate satisfies
the bounds strictly. One step, with `g = M⁻¹ ∂F/∂z` (see [`riesz!`](@ref)):

    ψ ← ψ - α (g + μ M⁻¹a),

where `μ ≥ 0` is zero if the constraint `aᵀz ≤ V` is inactive and otherwise solves `aᵀz(ψ) = V` (Illinois method).

Step size: `α₀ = 1/maximum(abs, g)` after `init`/`reset!`; afterwards `√(α_GBB α_prev)` with the generalised
Barzilai–Borwein step `α_GBB = (ψₖ - ψₖ₋₁, zₖ - zₖ₋₁)_M / |(gₖ - gₖ₋₁, zₖ - zₖ₋₁)_M|`. The step is halved until

- `linesearch = :armijo`: `F(z_new) ≤ F(z) + c1 ∂F/∂z ⋅ (z_new - z)`, or
- `linesearch = :bregman`: `F(z_new) ≤ F(z) + ∂F/∂z ⋅ (z_new - z) + D(z_new, z)/α`, with `D` the Bregman divergence
  of the (scaled) Fermi–Dirac entropy in the metric `M`.

A trial with a non-finite objective counts as failed. If no step is accepted after `max_backtracks` halvings, `z` is
left unchanged and `accepted = false`.

Optimality measure (`kkt` in [`StepInfo`](@ref)): `(1, η)_M` with `λ = (ψ_new - ψ)/α` and
`η = max(-(z - lb) λ, (ub - z) λ)` (eq. 23a of arXiv:2411.19421).
"""
struct SiMPL <: Optimizer
  linesearch::Symbol
  c1::Float64
  max_backtracks::Int
  root_tol::Float64
  max_root_iterations::Int
end

function SiMPL(; linesearch = :armijo, c1 = 1e-4, max_backtracks = 30, root_tol = 1e-12, max_root_iterations = 100)
  linesearch in (:armijo, :bregman) || throw(ArgumentError("linesearch must be :armijo or :bregman"))
  0 < c1 < 1 || throw(ArgumentError("c1 must be in (0, 1)"))
  SiMPL(linesearch, c1, max_backtracks, root_tol, max_root_iterations)
end

mutable struct SiMPLState{V}
  ψ::V; z::V; dF::V; g::V; ga::V
  ψ_prev::V; z_prev::V; g_prev::V
  ψ_trial::V; ψ_half::V; z_trial::V; work::V; one::V
  ψ_back::V; ψ_prev_back::V; z_prev_back::V; g_prev_back::V
  k::Int; α_prev::Float64; scale::Float64
  k_back::Int; α_prev_back::Float64
  resync::Bool
end

_σ(x) = inv(one(x) + exp(-x))
_softplus(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
_logσ(x) = -_softplus(-x)
_logit(s) = log(s) - log1p(-s)

function _to_latent!(ψ, z, lb, ub)
  ϵ = 1e-12
  @. ψ = _logit(clamp((z - lb) / (ub - lb), ϵ, 1 - ϵ))
end

_to_design!(z, ψ, lb, ub) = (@. z = lb + (ub - lb) * _σ(ψ))

function init(opt::SiMPL, z, oracle)
  v() = zero(similar(z))
  s = SiMPLState(v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(),
                 0, NaN, 1.0, 0, NaN, true)
  fill!(s.one, 1)
  _sync!(s, z, oracle)
  s
end

function _sync!(s::SiMPLState, z, oracle)
  lb, ub = bounds(oracle)
  _to_latent!(s.ψ, z, lb, ub)
  _to_design!(s.z, s.ψ, lb, ub)
  c = linear_constraint(oracle)
  c === nothing ? fill!(s.ga, 0) : riesz!(s.ga, oracle, c[1])
  s.resync = false
  s
end

function reset!(opt::SiMPL, s::SiMPLState)
  s.k = 0
  s.α_prev = NaN
  s.scale = 1.0
  s.resync = true   # the latent variable is rebuilt from z at the next step
  s
end

function shrink!(opt::SiMPL, s::SiMPLState, factor)
  copyto!(s.ψ, s.ψ_back); copyto!(s.ψ_prev, s.ψ_prev_back)
  copyto!(s.z_prev, s.z_prev_back); copyto!(s.g_prev, s.g_prev_back)
  s.k, s.α_prev = s.k_back, s.α_prev_back
  s.scale *= factor
  s.resync = true   # z was restored by the loop; rebuild the design copy from it
  s
end

# aᵀz(ψ_half - α μ ga) - V
function _constraint_gap(s, a, V, α, μ, lb, ub)
  @. s.ψ_trial = s.ψ_half - α * μ * s.ga
  _to_design!(s.z_trial, s.ψ_trial, lb, ub)
  dot(a, s.z_trial) - V
end

function _volume_shift(opt, s, a, V, α, lb, ub)
  h0 = _constraint_gap(s, a, V, α, 0.0, lb, ub)
  h0 <= 0 && return 0.0
  μ0, h_lo = 0.0, h0
  μ1 = max(maximum(abs, s.g), 1.0)
  h_hi = _constraint_gap(s, a, V, α, μ1, lb, ub)
  n = 0
  while h_hi > 0 && n < opt.max_root_iterations
    μ0, h_lo = μ1, h_hi
    μ1 *= 2
    h_hi = _constraint_gap(s, a, V, α, μ1, lb, ub)
    n += 1
  end
  h_hi > 0 && error("SiMPL: could not bracket the multiplier of the linear constraint")
  # Illinois method on [μ0, μ1], h(μ0) > 0 > h(μ1)
  side = 0
  μ = μ1
  for _ in 1:opt.max_root_iterations
    μ = (μ0 * h_hi - μ1 * h_lo) / (h_hi - h_lo)
    h = _constraint_gap(s, a, V, α, μ, lb, ub)
    abs(h) <= opt.root_tol * max(abs(V), 1) && break
    if h > 0
      μ0, h_lo = μ, h
      side == -1 && (h_hi /= 2)
      side = -1
    else
      μ1, h_hi = μ, h
      side == 1 && (h_lo /= 2)
      side = 1
    end
    abs(μ1 - μ0) <= eps(μ1) * 4 && break
  end
  μ
end

function step!(opt::SiMPL, s::SiMPLState, z, oracle, ctx)
  s.resync && _sync!(s, z, oracle)
  lb, ub = bounds(oracle)
  c = linear_constraint(oracle)

  f = objective!(s.dF, oracle, s.z)
  nev = 1
  riesz!(s.g, oracle, s.dF)

  if s.k == 0 || !isfinite(s.α_prev)
    gmax = maximum(abs, s.g)
    α = gmax > 0 ? 1 / gmax : 1.0
  else
    @. s.work = s.z - s.z_prev
    @. s.ψ_trial = s.ψ - s.ψ_prev
    num = inner(oracle, s.ψ_trial, s.work)
    @. s.ψ_trial = s.g - s.g_prev
    den = abs(inner(oracle, s.ψ_trial, s.work))
    α_gbb = (den > 0 && num > 0) ? num / den : s.α_prev
    α = sqrt(α_gbb * s.α_prev)
  end
  α *= s.scale

  accepted = false
  f_new = NaN
  for _ in 0:opt.max_backtracks
    @. s.ψ_half = s.ψ - α * s.g
    if c === nothing
      copyto!(s.ψ_trial, s.ψ_half)
      _to_design!(s.z_trial, s.ψ_trial, lb, ub)
    else
      μ = _volume_shift(opt, s, c[1], c[2], α, lb, ub)
      _constraint_gap(s, c[1], c[2], α, μ, lb, ub)   # leaves ψ_trial and z_trial at μ
    end
    f_new = objective(oracle, s.z_trial)
    nev += 1
    if isfinite(f_new)
      @. s.work = s.z_trial - s.z
      decrease = dot(s.dF, s.work)
      bound = if opt.linesearch === :armijo
        f + opt.c1 * decrease
      else
        f + decrease + _bregman(oracle, s, lb, ub) / α
      end
      if f_new <= bound
        accepted = true
        break
      end
    end
    α /= 2
  end

  if !accepted
    return StepInfo(false, f, f_new, NaN, 0.0, α, nev)
  end

  # optimality measure at the current design
  @. s.work = (s.ψ_trial - s.ψ) / α
  @. s.work = max(-(s.z - lb) * s.work, (ub - s.z) * s.work)
  kkt = inner(oracle, s.one, s.work)

  # memory for shrink!
  copyto!(s.ψ_back, s.ψ); copyto!(s.ψ_prev_back, s.ψ_prev)
  copyto!(s.z_prev_back, s.z_prev); copyto!(s.g_prev_back, s.g_prev)
  s.k_back, s.α_prev_back = s.k, s.α_prev

  copyto!(s.ψ_prev, s.ψ); copyto!(s.z_prev, s.z); copyto!(s.g_prev, s.g)
  copyto!(s.ψ, s.ψ_trial); copyto!(s.z, s.z_trial)
  s.α_prev = α
  s.k += 1
  s.scale = 1.0

  @. s.work = s.z - s.z_prev
  change = maximum(abs, s.work)
  copyto!(z, s.z)
  StepInfo(true, f, f_new, kkt, change, α, nev)
end

# (1, D(z_trial, z))_M for the scaled Fermi–Dirac entropy; uses the latent variables for stability
function _bregman(oracle, s, lb, ub)
  @. s.work = (ub - lb) * (_σ(s.ψ_trial) * (_logσ(s.ψ_trial) - _logσ(s.ψ)) +
                           _σ(-s.ψ_trial) * (_logσ(-s.ψ_trial) - _logσ(-s.ψ)))
  inner(oracle, s.one, s.work)
end
