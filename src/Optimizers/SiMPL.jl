"""
    SiMPL(; linesearch = :armijo, c1 = 1e-4, max_backtracks = 30, max_growth = 10.0, root_tol = 1e-12,
            max_root_iterations = 100, constraint_tol = 1e-6, max_corrections = 8)

Sigmoidal mirror descent with a projected latent variable (Keith, Kim, Lazarov and Surowiec, arXiv:2409.19341;
Kim, Lazarov, Surowiec and Keith, arXiv:2411.19421).

The design is written as `z = lb + (ub - lb) σ(ψ)` with the sigmoid `σ(x) = 1/(1 + e⁻ˣ)`, so every iterate satisfies
the bounds strictly. One step, with `g = M⁻¹ ∂F/∂z` (see [`riesz!`](@ref)):

    ψ ← ψ - α (g + μ M⁻¹a),

where `μ ≥ 0` is zero if the constraint `aᵀz ≤ V` is inactive and otherwise solves `aᵀz(ψ) = V` (Newton iteration
kept inside a bracket).

Step size: `α₀ = 1/maximum(abs, g)` after `init`/`reset!`; afterwards `√(α_GBB α_prev)` with the generalised
Barzilai–Borwein step `α_GBB = (ψₖ - ψₖ₋₁, zₖ - zₖ₋₁)_M / |(gₖ - gₖ₋₁, zₖ - zₖ₋₁)_M|`, capped at
`max_growth * α_prev` (each rejected trial costs a forward solve). The step is halved until

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

Nonlinear constraint `g(z) ≤ 0` (for example the volume of a projected density; an extension of the method, which
the references only state for a linear volume): at each step `a = ∂g/∂z` at the current design (floored at
`1e-12 maximum(a)`) and the linearised constraint `aᵀz ≤ aᵀzₖ - g(zₖ) - s` takes the place of `aᵀz ≤ V`. For each
trial, the shift `s` is corrected (chord, then bounded secant steps, at most `max_corrections`, one evaluation of the
true `g` each) until `-constraint_tol ≤ g(z_trial) ≤ 0`, aiming at `-constraint_tol/2`, so every accepted iterate is
feasible; `s` is kept between steps (`reset!` clears it) and divided by 4 each time the line search halves the step
(the linearisation error falls with the square of the step). A tolerance much larger than `1e-6` can stall the line
search, because the slack of the constraint then moves the objective more than the expected decrease. At an
infeasible design (`g > constraint_tol`: a start or a continuation jump) the step restores feasibility: the first
trial with a finite objective is accepted, without the Armijo test.

It also reports the stationarity measure of the design before the step ([`stationarity!`](@ref)), the criterion of the
paper.
"""
struct SiMPL <: Optimizer
  linesearch::Symbol
  c1::Float64
  max_backtracks::Int
  max_growth::Float64
  root_tol::Float64
  max_root_iterations::Int
  constraint_tol::Float64
  max_corrections::Int
end

function SiMPL(; linesearch = :armijo, c1 = 1e-4, max_backtracks = 30, max_growth = 10.0, root_tol = 1e-12,
               max_root_iterations = 100, constraint_tol = 1e-6, max_corrections = 8)
  linesearch in (:armijo, :bregman) || throw(ArgumentError("linesearch must be :armijo or :bregman"))
  0 < c1 < 1 || throw(ArgumentError("c1 must be in (0, 1)"))
  max_growth >= 1 || throw(ArgumentError("max_growth must be at least 1"))
  0 < constraint_tol || throw(ArgumentError("constraint_tol must be positive"))
  max_corrections >= 0 || throw(ArgumentError("max_corrections must be non-negative"))
  SiMPL(linesearch, c1, max_backtracks, max_growth, root_tol, max_root_iterations, constraint_tol, max_corrections)
end

mutable struct SiMPLState{V}
  ψ::V; z::V; dF::V; g::V; ga::V          # current latent variable, design, derivative, gradient, M⁻¹a
  ψ_prev::V; z_prev::V; g_prev::V         # previous iterate (Barzilai–Borwein)
  ψ_trial::V; z_trial::V; work::V; one::V
  a::V                   # derivative of a nonlinear constraint at z
  k::Int
  μ::Float64             # last multiplier of the linear constraint (starting guess of the next search)
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
  shift::Float64         # correction of the linearised nonlinear constraint, kept between steps
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
  s = SiMPLState(v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(),
                 0, 0.0, NaN, NaN, NaN, NaN, NaN, 1.0, true, false, false, false, false, false, 0.0)
  fill!(s.one, 1)
  _sync!(s, z, oracle)
  s
end

function _sync!(s::SiMPLState, z, oracle)
  lb, ub = bounds(oracle)
  _to_latent!(s.ψ, z, lb, ub)
  _to_design!(s.z, s.ψ, lb, ub)
  num_constraints(oracle) <= 1 ||
    throw(ArgumentError("SiMPL treats one constraint itself; the oracle has $(num_constraints(oracle))"))
  (num_constraints(oracle) == 0 || handles(SiMPL(), constraint_info(oracle, 1))) ||
    throw(ArgumentError("SiMPL treats one constraint g(z) ≤ 0 that solves no state; the oracle gives $(constraint_info(oracle, 1))"))
  c = num_constraints(oracle) == 0 ? nothing : linear_constraint(oracle, 1)
  if c === nothing
    fill!(s.ga, 0)   # a nonlinear constraint sets it at every step
  else
    (minimum(c[1]) >= 0 && maximum(c[1]) > 0) ||
      throw(ArgumentError("SiMPL needs a linear constraint with a ≥ 0, not all zero"))
    riesz!(s.ga, oracle, c[1])
  end
  s.resync = false
  s.μ = 0.0
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
  s.shift = 0.0
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
  s.shift *= factor^2   # the linearisation error falls with the square of the step
  s.shrunk = true
  s
end

constraint_tol(opt::SiMPL) = opt.constraint_tol

# Sets ψ_trial = ψ - α (g + μ ga) and z_trial
function _set_trial!(s, α, μ, lb, ub)
  @. s.ψ_trial = s.ψ - α * (s.g + μ * s.ga)
  _to_design!(s.z_trial, s.ψ_trial, lb, ub)
  nothing
end

_gap!(s, a, V, α, μ, lb, ub) = (_set_trial!(s, α, μ, lb, ub); dot(a, s.z_trial) - V)

# Multiplier of the linear constraint for the step α: root of h(μ) = aᵀz(ψ - α(g + μ ga)) - V, which decreases with μ.
# Newton iteration (the derivative of the sigmoid is analytical) kept inside a bracket, with bisection as fallback,
# started from the multiplier of the previous call. The target is h = -tol/2 and the result satisfies -tol ≤ h ≤ 0,
# so every iterate is feasible. Leaves ψ_trial and z_trial at the returned multiplier.
function _volume_shift!(opt, s, a, V, α, lb, ub)
  h = _gap!(s, a, V, α, 0.0, lb, ub)
  h <= 0 && return 0.0
  half = opt.root_tol * max(abs(V), 1.0) / 2
  h += half
  μ, μ0, μ1 = 0.0, 0.0, Inf   # h(μ0) > 0 ≥ h(μ1), with h shifted by tol/2
  if s.μ > 0
    μ = s.μ
    h = _gap!(s, a, V, α, μ, lb, ub) + half
    abs(h) <= half && return μ
    h > 0 ? (μ0 = μ) : (μ1 = μ)
  end
  for _ in 1:opt.max_root_iterations
    @. s.work = s.ga * (s.z_trial - lb) * (ub - s.z_trial) / (ub - lb)
    dh = -α * dot(a, s.work)
    μn = dh < 0 ? μ - h / dh : Inf
    if !(μ0 < μn < μ1)   # Newton left the bracket, or the slope vanished
      μn = isfinite(μ1) ? (μ0 + μ1) / 2 : max(2 * μ0, maximum(abs, s.g), 1.0)
    end
    μn == μ && break
    μ = μn
    h = _gap!(s, a, V, α, μ, lb, ub) + half
    if abs(h) <= half
      s.μ = μ
      return μ
    end
    h > 0 ? (μ0 = μ) : (μ1 = μ)
    μ1 - μ0 <= 4 * eps(μ1) && break
  end
  isfinite(μ1) || error("SiMPL: could not bracket the multiplier of the linear constraint (is aᵀlb ≤ V?)")
  if h > half   # stopped on the infeasible side: take the feasible end of the bracket
    μ = μ1
    _gap!(s, a, V, α, μ, lb, ub)
  end
  s.μ = μ
  μ
end

_violation(c::Nothing, z) = 0.0
_violation(c, z) = max(0.0, dot(c[1], z) - c[2])

function step!(opt::SiMPL, s::SiMPLState, z, oracle, ctx)
  s.resync && _sync!(s, z, oracle)
  s.shrunk || (s.scale = 1.0)   # the previous step stood
  s.shrunk = false
  lb, ub = bounds(oracle)
  nc = num_constraints(oracle)
  c = nc == 0 ? nothing : linear_constraint(oracle, 1)
  nev = 0

  if !s.have_grad
    if !s.have_f
      s.f = objective(oracle, s.z)
      nev += 1
      s.have_f = true
    end
    isfinite(s.f) || return StepInfo(false, s.f, NaN, NaN, 0.0, 0.0, _violation(c, s.z), nev, NaN, NaN)
    gradient!(s.dF, oracle, s.z)
    riesz!(s.g, oracle, s.dF)
    s.have_grad = true
  end
  f = s.f

  # nonlinear constraint: linearised at z (aᵀz ≤ Vlin, then corrected per trial)
  nonlinear = c === nothing && nc == 1
  gk, Vlin, restoration = NaN, NaN, false
  if nonlinear
    gk = constraint(oracle, 1, s.z)
    isfinite(gk) || return StepInfo(false, f, NaN, NaN, 0.0, 0.0, Inf, nev, NaN, NaN)
    constraint_gradient!(s.a, oracle, 1, s.z)
    amax = maximum(s.a)
    amax > 0 || throw(ArgumentError("SiMPL: the derivative of the constraint has no positive entry"))
    @. s.a = max(s.a, 1e-12 * amax)
    riesz!(s.ga, oracle, s.a)
    Vlin = dot(s.a, s.z) - gk
    alb = lb isa Number ? lb * sum(s.a) : dot(s.a, lb)   # aᵀlb: the linearised target must stay above it
    restoration = gk > opt.constraint_tol   # infeasible design (start, continuation jump): feasibility first
  end
  shift0 = s.shift
  violation_k = nonlinear ? max(0.0, gk) : _violation(c, s.z)
  # stationarity measure at z (the projection uses the constraint linearised at z)
  stat = nonlinear ? stationarity!(s.work, oracle, s.z, s.g, s.ga, s.a, Vlin) :
         c === nothing ? stationarity!(s.work, oracle, s.z, s.g, s.ga, nothing, NaN) :
                         stationarity!(s.work, oracle, s.z, s.g, s.ga, c[1], c[2])

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
    guess = min(sqrt(α_gbb * s.α_prev), opt.max_growth * s.α_prev)
  end
  s.α_guess = guess
  α = guess * s.scale

  slack = 10 * eps(Float64) * abs(f)
  μ = (c === nothing && !nonlinear) ? NaN : 0.0
  G = gk
  accepted = false
  f_new = NaN
  kkt = NaN
  change = 0.0
  for j in 0:opt.max_backtracks
    if nonlinear
      μ, G = _corrected_shift!(opt, s, oracle, Vlin, alb, α, lb, ub)
    elseif c === nothing
      _set_trial!(s, α, 0.0, lb, ub)
    else
      μ = _volume_shift!(opt, s, c[1], c[2], α, lb, ub)
    end
    if j == 0   # optimality measure at the current design
      @. s.work = (s.ψ_trial - s.ψ) / α
      @. s.work = max(-(s.z - lb) * s.work, (ub - s.z) * s.work)
      kkt = inner(oracle, s.one, s.work)
    end
    @. s.work = s.z_trial - s.z
    change = maximum(abs, s.work)
    if change == 0   # nothing left to move at this precision
      return StepInfo(true, f, f, kkt, 0.0, α, violation_k, nev, μ, stat)
    end
    decrease = dot(s.dF, s.work)
    f_new = objective(oracle, s.z_trial)
    nev += 1
    if isfinite(f_new) && restoration && isfinite(G) && G < gk
      accepted = true   # restoration step: the first trial with a finite objective that reduces the violation
      break
    end
    # a nonlinear constraint that could not be met at the trial (corrections exhausted) makes the trial fail
    if isfinite(f_new) && !restoration && (!nonlinear || (isfinite(G) && G <= 0))
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
    nonlinear && (s.shift /= 4)   # the linearisation error, and so the shift, falls with α²
  end

  # the forward state of the oracle is now at a rejected trial; f, dF and g at z remain valid
  accepted || (s.shift = shift0; return StepInfo(false, f, f_new, kkt, 0.0, α, violation_k, nev, μ, stat))

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
  StepInfo(true, f, f_new, kkt, change, α, nonlinear ? max(0.0, G) : _violation(c, s.z), nev, μ, stat)
end

# Multiplier of the linearised constraint aᵀz ≤ Vlin - s for the step α, with the shift s corrected until the true
# constraint at the trial satisfies -tol ≤ g ≤ 0 (or holds with μ = 0), aiming at -tol/2. Returns (μ, g(z_trial)) and
# leaves ψ_trial and z_trial at the returned multiplier.
function _corrected_shift!(opt, s, oracle, Vlin, alb, α, lb, ub)
  tol = opt.constraint_tol
  # the target aᵀz = Vlin - s must stay above aᵀlb, or the multiplier has no root (the trial then goes to the bounds)
  smax = Vlin - alb - 1e-12 * max(abs(Vlin), 1.0)
  cap(x) = isfinite(x) ? min(x, smax) : smax
  sh = cap(s.shift)
  μ = _volume_shift!(opt, s, s.a, Vlin - sh, α, lb, ub)
  G = constraint(oracle, 1, s.z_trial)
  lo, hi, prev = (-Inf, NaN), (Inf, NaN), (NaN, NaN)
  for _ in 1:opt.max_corrections
    isfinite(G) || break
    (G <= 0 && (μ == 0 || G >= -tol)) && break
    t = G + tol / 2
    G > 0 ? (lo = (sh, t)) : (hi = (sh, t))
    sn = cap(_next_shift(sh, t, lo, hi, prev))
    sn == sh && break   # pinned at the bounds: no correction can do more
    prev = (sh, t)
    sh = sn
    μ = _volume_shift!(opt, s, s.a, Vlin - sh, α, lb, ub)
    G = constraint(oracle, 1, s.z_trial)
  end
  s.shift = isfinite(sh) ? sh : 0.0
  μ, G
end

# (1, D(z_trial, z))_M for the scaled Fermi–Dirac entropy; uses the latent variables for stability
function _bregman(oracle, s, lb, ub)
  @. s.work = (ub - lb) * (_σ(s.ψ_trial) * (_logσ(s.ψ_trial) - _logσ(s.ψ)) +
                           _σ(-s.ψ_trial) * (_logσ(-s.ψ_trial) - _logσ(-s.ψ)))
  inner(oracle, s.one, s.work)
end

# OC and SiMPL treat one inequality g ≤ 0 that solves no state (each correction of the step evaluates it)
handles(::Union{OC,SiMPL}, c::ConstraintInfo) = is_upper(c) && c.hi == 0 && c.design_only && !c.field
max_native(::Union{OC,SiMPL}) = 1
requires_native(::OC) = true

# memory for a checkpoint: the latent variable (not recoverable from z, which init clamps 1e-12 away from the bounds),
# the previous iterate and gradient of Barzilai–Borwein, the step, the multiplier and the correction shift. The value
# and gradient at z are evaluated again after a restore.
memory(::SiMPL, s::SiMPLState, z) = (k = Float64(s.k), μ = s.μ, α_prev = s.α_prev, shift = s.shift,
                                     ψ = s.ψ, ψ_prev = s.ψ_prev, z_prev = s.z_prev, g_prev = s.g_prev)

function restore_memory!(::SiMPL, s::SiMPLState, z, v::NamedTuple)
  _check_memory(v, (:k, :μ, :α_prev, :shift, :ψ, :ψ_prev, :z_prev, :g_prev), length(z), "SiMPL")
  s.k = Int(v.k)
  s.μ = v.μ
  s.α_prev = v.α_prev
  s.shift = v.shift
  copyto!(s.ψ, v.ψ)
  copyto!(s.ψ_prev, v.ψ_prev)
  copyto!(s.z_prev, v.z_prev)
  copyto!(s.g_prev, v.g_prev)
  copyto!(s.z, z)
  s.resync = false
  s.have_f = false
  s.have_grad = false
  s.can_rollback = false
  s.retry = false
  s.shrunk = false
  s.scale = 1.0
  s
end
