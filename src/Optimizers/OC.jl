"""
    OC(; move = 0.2, damping = 0.5, bisection_tol = 1e-3, max_bisections = 200)

Optimality criteria update of Andreassen et al. (top88, 2011) for one linear constraint `aᵀz ≤ V`:

    zᵢ ← clamp(zᵢ Bᵢ^damping, max(lbᵢ, zᵢ - m), min(ubᵢ, zᵢ + m)),   Bᵢ = max(0, -∂F/∂zᵢ) / (λ aᵢ),

with `m = move * (ub - lb)` and `λ > 0` found by bisection on `aᵀz = V`, starting from `[0, 1e9]`. The bisection
stops when `(λ₂ - λ₁)/(λ₁ + λ₂) ≤ bisection_tol` and, as in top88, keeps the design of the last midpoint, so the
constraint holds up to the bisection tolerance. With the defaults, the step is the one of top88.

Requires a linear constraint with `a > 0`. The objective is evaluated once per step and the new design is not
evaluated (`f_new = NaN`, `kkt = NaN`). If the move limit does not allow reaching `aᵀz ≤ V` (infeasible start), the
step is still taken and `violation > 0` in the [`StepInfo`](@ref).
"""
struct OC <: Optimizer
  move::Float64
  damping::Float64
  bisection_tol::Float64
  max_bisections::Int
end

function OC(; move = 0.2, damping = 0.5, bisection_tol = 1e-3, max_bisections = 200)
  0 < move || throw(ArgumentError("move must be positive"))
  0 < damping || throw(ArgumentError("damping must be positive"))
  OC(move, damping, bisection_tol, max_bisections)
end

mutable struct OCState{V}
  move::Float64
  shrunk::Bool   # shrink! was called since the last step!
  dF::V
  z_old::V
end

function init(opt::OC, z, oracle)
  c = linear_constraint(oracle)
  c === nothing && throw(ArgumentError("OC needs a linear constraint aᵀz ≤ V"))
  minimum(c[1]) > 0 || throw(ArgumentError("OC needs a linear constraint with a > 0"))
  OCState(opt.move, false, similar(z), similar(z))
end

reset!(opt::OC, s::OCState) = (s.move = opt.move; s.shrunk = false; s)
shrink!(opt::OC, s::OCState, factor) = (s.move *= factor; s.shrunk = true; s)

_oc_power(x, η) = η == 0.5 ? sqrt(x) : x^η

# B holds (max(0, -∂F/∂z) / a)^η and κ = λ^(-η), so the power is not recomputed at every bisection
function _oc_update!(z, zo, B, a, lb, ub, κ, m)
  @. z = clamp(zo * B * κ, max(lb, zo - m * (ub - lb)), min(ub, zo + m * (ub - lb)))
  dot(a, z)
end

function step!(opt::OC, s::OCState, z, oracle, ctx)
  s.shrunk || (s.move = opt.move)   # the previous step stood: the reduction is dropped
  s.shrunk = false
  f = objective(oracle, z)
  isfinite(f) || return StepInfo(false, f, NaN, NaN, 0.0, s.move, 0.0, 1)
  gradient!(s.dF, oracle, z)
  a, V = linear_constraint(oracle)
  lb, ub = bounds(oracle)
  copyto!(s.z_old, z)
  zo, B, η, m = s.z_old, s.dF, opt.damping, s.move
  @. B = _oc_power(max(zero(B), -B) / a, η)

  λ1, λ2 = 0.0, 1e9
  n = 0
  vol = _oc_update!(z, zo, B, a, lb, ub, λ2^(-η), m)
  while vol > V && n < 60   # widen the bracket if the objective is badly scaled
    λ1, λ2 = λ2, 10λ2
    vol = _oc_update!(z, zo, B, a, lb, ub, λ2^(-η), m)
    n += 1
  end
  n = 0
  while (λ2 - λ1) / (λ1 + λ2) > opt.bisection_tol && n < opt.max_bisections
    λ = (λ1 + λ2) / 2
    vol = _oc_update!(z, zo, B, a, lb, ub, λ^(-η), m)   # as in top88, z keeps the update of the last midpoint
    vol > V ? (λ1 = λ) : (λ2 = λ)
    n += 1
  end

  @. B = z - zo   # B is free now: use it as work vector
  change = maximum(abs, B)
  StepInfo(true, f, NaN, NaN, change, m, max(0.0, vol - V), 1)
end
