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
left unchanged and `accepted = false`. The tests allow a decrease lost in rounding (`10 eps |F|`), so a converged
design keeps returning accepted steps with a negligible change instead of a failed line search.

Cost per step: one call to `gradient!` and one call to `objective` per trial (the value at the accepted trial is
reused as the value of the next step). After `init`, `reset!` or a change of `z` by the caller, one more `objective`.

Optimality measure (`kkt` in [`StepInfo`](@ref)): `(1, η)_M` with `λ = (ψ_trial - ψ)/α` at the first trial of the
step and `η = max(-(z - lb) λ, (ub - z) λ)` (eq. 23a of arXiv:2411.19421).

Requires `a ≥ 0` (not all zero) in the linear constraint.
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
  ψ::V; z::V; dF::V; g::V; ga::V          # current latent variable, design, derivative, gradient, M⁻¹a
  ψ_prev::V; z_prev::V; g_prev::V         # previous iterate (Barzilai–Borwein)
  ψ_trial::V; z_trial::V; work::V; one::V
  k::Int
  α_prev::Float64        # accepted step of the previous iteration
  α_prev_back::Float64   # its value before the last accepted step
  α_guess::Float64       # initial guess of the last step, before scaling
  f::Float64             # objective at z
  f_back::Float64        # objective before the last accepted step
  scale::Float64         # reduction asked by shrink!
  resync::Bool           # rebuild ψ from the caller's z
  have_f::Bool
  have_grad::Bool
  shrunk::Bool           # shrink! was called since the last step!
  can_rollback::Bool
  retry::Bool            # the next step repeats a rolled-back step
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
  v() = similar(z)
  s = SiMPLState(v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(),
                 0, NaN, NaN, NaN, NaN, NaN, 1.0, true, false, false, false, false, false)
  fill!(s.one, 1)
  _sync!(s, z, oracle)
  s
end

function _sync!(s::SiMPLState, z, oracle)
  lb, ub = bounds(oracle)
  _to_latent!(s.ψ, z, lb, ub)
  _to_design!(s.z, s.ψ, lb, ub)
  c = linear_constraint(oracle)
  if c === nothing
    fill!(s.ga, 0)
  else
    (minimum(c[1]) >= 0 && maximum(c[1]) > 0) ||
      throw(ArgumentError("SiMPL needs a linear constraint with a ≥ 0, not all zero"))
    riesz!(s.ga, oracle, c[1])
  end
  s.resync = false
  s.have_f = false
  s.have_grad = false
  s
end

function reset!(opt::SiMPL, s::SiMPLState)
  s.k = 0
  s.α_prev = NaN
  s.scale = 1.0
  s.shrunk = false
  s.can_rollback = false
  s.retry = false
  s.resync = true   # the latent variable is rebuilt from z at the next step
  s
end

function shrink!(opt::SiMPL, s::SiMPLState, factor)
  if s.can_rollback
    # back to the start of the last accepted step; its derivative is still in dF and its gradient in g_prev
    s.ψ, s.ψ_prev = s.ψ_prev, s.ψ
    s.z, s.z_prev = s.z_prev, s.z
    s.g, s.g_prev = s.g_prev, s.g
    s.k -= 1
    s.α_prev = s.α_prev_back
    s.f = s.f_back
    s.have_f = true
    s.have_grad = true
    s.retry = true
    s.can_rollback = false
  end
  s.scale *= factor
  s.shrunk = true
  s
end

# Sets ψ_trial = ψ - α (g + μ ga) and z_trial
function _set_trial!(s, α, μ, lb, ub)
  @. s.ψ_trial = s.ψ - α * (s.g + μ * s.ga)
  _to_design!(s.z_trial, s.ψ_trial, lb, ub)
  nothing
end

_gap!(s, a, V, α, μ, lb, ub) = (_set_trial!(s, α, μ, lb, ub); dot(a, s.z_trial) - V)

# Multiplier of the linear constraint for the step α. Leaves ψ_trial and z_trial at the returned multiplier.
function _volume_shift!(opt, s, a, V, α, lb, ub)
  h = _gap!(s, a, V, α, 0.0, lb, ub)
  h <= 0 && return 0.0
  tol = opt.root_tol * max(abs(V), 1.0)
  μ0, h0 = 0.0, h                               # h0 > 0
  μ1 = max(maximum(abs, s.g), 1.0)
  h1 = _gap!(s, a, V, α, μ1, lb, ub)            # h1 ≤ 0 once bracketed
  n = 0
  while h1 > 0 && n < opt.max_root_iterations
    μ0, h0 = μ1, h1
    μ1 *= 2
    h1 = _gap!(s, a, V, α, μ1, lb, ub)
    n += 1
  end
  h1 > 0 && error("SiMPL: could not bracket the multiplier of the linear constraint (is aᵀlb ≤ V?)")
  # Illinois method on [μ0, μ1]
  μ, h, side = μ1, h1, 0
  for _ in 1:opt.max_root_iterations
    abs(h) <= tol && break
    μ1 - μ0 <= 4 * eps(μ1) && break
    μ = (μ0 * h1 - μ1 * h0) / (h1 - h0)
    h = _gap!(s, a, V, α, μ, lb, ub)
    if h > 0
      μ0, h0 = μ, h
      side == -1 && (h1 /= 2)
      side = -1
    else
      μ1, h1 = μ, h
      side == 1 && (h0 /= 2)
      side = 1
    end
  end
  if h > tol   # stopped on the infeasible side: take the feasible end of the bracket
    μ = μ1
    _gap!(s, a, V, α, μ, lb, ub)
  end
  μ
end

_violation(c::Nothing, z) = 0.0
_violation(c, z) = max(0.0, dot(c[1], z) - c[2])

function step!(opt::SiMPL, s::SiMPLState, z, oracle, ctx)
  s.resync && _sync!(s, z, oracle)
  s.shrunk || (s.scale = 1.0)   # the previous step stood
  s.shrunk = false
  lb, ub = bounds(oracle)
  c = linear_constraint(oracle)
  nev = 0

  if !s.have_grad
    if !s.have_f
      s.f = objective(oracle, s.z)
      nev += 1
      s.have_f = true
    end
    isfinite(s.f) || return StepInfo(false, s.f, NaN, NaN, 0.0, 0.0, _violation(c, s.z), nev)
    gradient!(s.dF, oracle, s.z)
    riesz!(s.g, oracle, s.dF)
    s.have_grad = true
  end
  f = s.f

  if s.retry
    guess = s.α_guess
  elseif s.k == 0 || !isfinite(s.α_prev)
    gmax = maximum(abs, s.g)
    guess = gmax > 0 ? 1 / gmax : 1.0
  else
    @. s.work = s.z - s.z_prev
    @. s.ψ_trial = s.ψ - s.ψ_prev
    num = inner(oracle, s.ψ_trial, s.work)
    @. s.ψ_trial = s.g - s.g_prev
    den = abs(inner(oracle, s.ψ_trial, s.work))
    α_gbb = (den > 0 && num > 0) ? num / den : s.α_prev
    guess = sqrt(α_gbb * s.α_prev)
  end
  s.α_guess = guess
  α = guess * s.scale

  slack = 10 * eps(Float64) * abs(f)
  accepted = false
  f_new = NaN
  kkt = NaN
  change = 0.0
  for j in 0:opt.max_backtracks
    if c === nothing
      _set_trial!(s, α, 0.0, lb, ub)
    else
      _volume_shift!(opt, s, c[1], c[2], α, lb, ub)
    end
    if j == 0   # optimality measure at the current design
      @. s.work = (s.ψ_trial - s.ψ) / α
      @. s.work = max(-(s.z - lb) * s.work, (ub - s.z) * s.work)
      kkt = inner(oracle, s.one, s.work)
    end
    @. s.work = s.z_trial - s.z
    change = maximum(abs, s.work)
    if change == 0   # nothing left to move at this precision
      return StepInfo(true, f, f, kkt, 0.0, α, _violation(c, s.z), nev)
    end
    decrease = dot(s.dF, s.work)
    f_new = objective(oracle, s.z_trial)
    nev += 1
    if isfinite(f_new)
      bound = if opt.linesearch === :armijo
        f + opt.c1 * decrease
      else
        f + decrease + _bregman(oracle, s, lb, ub) / α
      end
      if f_new <= bound + slack
        accepted = true
        break
      end
    end
    α /= 2
  end

  # the forward state of the oracle is now at a rejected trial; f, dF and g at z remain valid
  accepted || return StepInfo(false, f, f_new, kkt, 0.0, α, _violation(c, s.z), nev)

  s.f_back, s.α_prev_back = f, s.α_prev
  s.ψ_prev, s.ψ, s.ψ_trial = s.ψ, s.ψ_trial, s.ψ_prev
  s.z_prev, s.z, s.z_trial = s.z, s.z_trial, s.z_prev
  s.g_prev, s.g = s.g, s.g_prev
  s.f = f_new
  s.have_f = true
  s.have_grad = false
  s.α_prev = α
  s.k += 1
  s.can_rollback = true
  s.retry = false

  copyto!(z, s.z)
  StepInfo(true, f, f_new, kkt, change, α, _violation(c, s.z), nev)
end

# (1, D(z_trial, z))_M for the scaled Fermi–Dirac entropy; uses the latent variables for stability
function _bregman(oracle, s, lb, ub)
  @. s.work = (ub - lb) * (_σ(s.ψ_trial) * (_logσ(s.ψ_trial) - _logσ(s.ψ)) +
                           _σ(-s.ψ_trial) * (_logσ(-s.ψ_trial) - _logσ(-s.ψ)))
  inner(oracle, s.one, s.work)
end
