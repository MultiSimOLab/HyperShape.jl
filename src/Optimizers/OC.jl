"""
    OC(; move = 0.2, damping = 0.5, bisection_tol = 1e-3, max_bisections = 200, constraint_tol = 1e-4,
         max_corrections = 8)

Optimality criteria update of Andreassen et al. (top88, 2011) for one constraint `g(z) ≤ 0`:

    zᵢ ← clamp(zᵢ Bᵢ^damping, max(lbᵢ, zᵢ - m), min(ubᵢ, zᵢ + m)),   Bᵢ = max(0, -∂F/∂zᵢ) / (λ aᵢ),

with `m = move * (ub - lb)` and `λ > 0` found by bisection on `aᵀz = V`, starting from `[0, 1e9]`.

- **Linear constraint** `aᵀz ≤ V`: the bisection stops when `(λ₂ - λ₁)/(λ₁ + λ₂) ≤ bisection_tol` and, as in top88,
  keeps the design of the last midpoint, so the constraint holds up to the bisection tolerance. With the defaults, the
  step is the one of top88. Requires `a > 0`.
- **Nonlinear constraint** (for example the volume of a projected density): `a = ∂g/∂z` at the current design (floored
  at `1e-12 maximum(a)`, since a filter can leave tiny or negative entries where a projection saturates) and the
  linearised constraint `aᵀz ≤ aᵀzₖ - g(zₖ) - s`. The shift `s` is then corrected (chord, then bounded secant steps,
  at most `max_corrections`) until `|g(z_new)| ≤ constraint_tol`, evaluating the true `g` once per correction; `s` is
  kept for the next step (`reset!` clears it). The bisection is solved to `min(bisection_tol, 1e-7)` so that the
  corrections see a smooth function. If the move limit does not allow meeting the constraint, the step is still taken.

The objective is evaluated once per step and the new design is not evaluated (`f_new = NaN`, `kkt = NaN`). If the
move limit does not allow meeting the constraint (infeasible start), the step is still taken and `violation > 0` in
the [`StepInfo`](@ref).
"""
struct OC <: Optimizer
  move::Float64
  damping::Float64
  bisection_tol::Float64
  max_bisections::Int
  constraint_tol::Float64
  max_corrections::Int
end

function OC(; move = 0.2, damping = 0.5, bisection_tol = 1e-3, max_bisections = 200, constraint_tol = 1e-4,
            max_corrections = 8)
  0 < move || throw(ArgumentError("move must be positive"))
  0 < damping || throw(ArgumentError("damping must be positive"))
  0 < constraint_tol || throw(ArgumentError("constraint_tol must be positive"))
  max_corrections >= 0 || throw(ArgumentError("max_corrections must be non-negative"))
  OC(move, damping, bisection_tol, max_bisections, constraint_tol, max_corrections)
end

mutable struct OCState{V}
  move::Float64
  shrunk::Bool   # shrink! was called since the last step!
  dF::V
  z_old::V
  a::V           # derivative of a nonlinear constraint at the current design
  shift::Float64 # correction of the linearised constraint, kept between steps
end

function init(opt::OC, z, oracle)
  has_constraint(oracle) || throw(ArgumentError("OC needs a constraint (aᵀz ≤ V or g(z) ≤ 0)"))
  c = linear_constraint(oracle)
  c === nothing || minimum(c[1]) > 0 || throw(ArgumentError("OC needs a linear constraint with a > 0"))
  OCState(opt.move, false, similar(z), similar(z), similar(z), 0.0)
end

reset!(opt::OC, s::OCState) = (s.move = opt.move; s.shrunk = false; s.shift = 0.0; s)
constraint_tol(opt::OC) = opt.constraint_tol
shrink!(opt::OC, s::OCState, factor) = (s.move *= factor; s.shrunk = true; s)

_oc_power(x, η) = η == 0.5 ? sqrt(x) : x^η

# B holds (max(0, -∂F/∂z) / a)^η and κ = λ^(-η), so the power is not recomputed at every bisection
function _oc_update!(z, zo, B, a, lb, ub, κ, m)
  @. z = clamp(zo * B * κ, max(lb, zo - m * (ub - lb)), min(ub, zo + m * (ub - lb)))
  dot(a, z)
end

# Bisection of λ on aᵀz = V; z keeps the update of the last midpoint (top88). Returns (aᵀz, λ, reached, active):
# `reached` is false when even λ = 1e69 leaves aᵀz > V (the move limit does not allow meeting the target), `active`
# is false when no λ tried left aᵀz > V (the constraint is inactive: λ → 0)
function _oc_bisect!(z, zo, B, a, lb, ub, V, m, η, tol, max_bisections)
  λ1, λ2 = 0.0, 1e9
  n = 0
  vol = _oc_update!(z, zo, B, a, lb, ub, λ2^(-η), m)
  while vol > V && n < 60   # widen the bracket if the objective is badly scaled
    λ1, λ2 = λ2, 10λ2
    vol = _oc_update!(z, zo, B, a, lb, ub, λ2^(-η), m)
    n += 1
  end
  reached = vol <= V
  n = 0
  λ = λ2
  while (λ2 - λ1) / (λ1 + λ2) > tol && n < max_bisections
    λ = (λ1 + λ2) / 2
    vol = _oc_update!(z, zo, B, a, lb, ub, λ^(-η), m)   # as in top88, z keeps the update of the last midpoint
    vol > V ? (λ1 = λ) : (λ2 = λ)
    n += 1
  end
  vol, λ, reached, λ1 > 0
end

function step!(opt::OC, s::OCState, z, oracle, ctx)
  s.shrunk || (s.move = opt.move)   # the previous step stood: the reduction is dropped
  s.shrunk = false
  f = objective(oracle, z)
  isfinite(f) || return StepInfo(false, f, NaN, NaN, 0.0, s.move, 0.0, 1, NaN)
  gradient!(s.dF, oracle, z)
  c = linear_constraint(oracle)
  c === nothing && return _oc_step_nonlinear!(opt, s, z, oracle, f)
  a, V = c
  lb, ub = bounds(oracle)
  copyto!(s.z_old, z)
  zo, B, η, m = s.z_old, s.dF, opt.damping, s.move
  @. B = _oc_power(max(zero(B), -B) / a, η)
  vol, λ, _, _ = _oc_bisect!(z, zo, B, a, lb, ub, V, m, η, opt.bisection_tol, opt.max_bisections)
  @. B = z - zo   # B is free now: use it as work vector
  change = maximum(abs, B)
  StepInfo(true, f, NaN, NaN, change, m, max(0.0, vol - V), 1, λ)
end

# Next shift of the linearised constraint from the corrections so far: regula falsi kept inside the bracket once the
# root is bracketed, secant (or the chord s + G) before
function _next_shift(sh, G, lo, hi, prev)
  slo, Glo = lo
  shi, Ghi = hi
  sp, Gp = prev
  if isfinite(slo) && isfinite(shi)
    sn = slo + (shi - slo) * Glo / (Glo - Ghi)
    return clamp(sn, slo + 0.05 * (shi - slo), shi - 0.05 * (shi - slo))
  elseif isfinite(sp) && (G - Gp) * (sh - sp) < 0
    return sh - G * (sh - sp) / (G - Gp)
  end
  sh + G
end

function _oc_step_nonlinear!(opt::OC, s::OCState, z, oracle, f)
  g = constraint(oracle, z)
  isfinite(g) || return StepInfo(false, f, NaN, NaN, 0.0, s.move, Inf, 1, NaN)
  a = constraint_gradient!(s.a, oracle, z)
  amax = maximum(a)
  amax > 0 || throw(ArgumentError("OC: the derivative of the constraint has no positive entry"))
  # an entry the constraint does not depend on at all has no OC update (Bᵢ would be infinite)
  any(iszero, a) && throw(ArgumentError("OC: the constraint does not depend on some entries of z (zero derivative); " *
                                        "OC needs a constraint that acts on every entry"))
  @. a = max(a, 1e-12 * amax)
  lb, ub = bounds(oracle)
  copyto!(s.z_old, z)
  zo, B, η, m = s.z_old, s.dF, opt.damping, s.move
  @. B = _oc_power(max(zero(B), -B) / a, η)
  tol = min(opt.bisection_tol, 1e-7)
  Vlin = dot(a, zo) - g
  sh = s.shift
  _, λ, reached, active = _oc_bisect!(z, zo, B, a, lb, ub, Vlin - sh, m, η, tol, opt.max_bisections)
  G = constraint(oracle, z)
  lo, hi, prev = (-Inf, NaN), (Inf, NaN), (NaN, NaN)
  for _ in 1:opt.max_corrections
    (isfinite(G) && abs(G) > opt.constraint_tol) || break
    (!reached && G > 0) && break   # the move limit does not allow getting closer in this step
    (!active && G < 0) && break    # inactive constraint: the step is the unconstrained one
    G > 0 ? (lo = (sh, G)) : (hi = (sh, G))
    sn = _next_shift(sh, G, lo, hi, prev)
    prev = (sh, G)
    sh = sn
    _, λ, reached, active = _oc_bisect!(z, zo, B, a, lb, ub, Vlin - sh, m, η, tol, opt.max_bisections)
    G = constraint(oracle, z)
  end
  s.shift = isfinite(sh) ? sh : 0.0
  @. B = z - zo
  StepInfo(true, f, NaN, NaN, maximum(abs, B), m, isfinite(G) ? max(0.0, G) : Inf, 1, λ)
end
