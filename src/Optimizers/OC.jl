"""
    OC(; move = 0.2, damping = 0.5, bisection_tol = 1e-3, max_bisections = 200)

Optimality criteria update of Andreassen et al. (top88, 2011) for one linear constraint `aᵀz ≤ V`:

    zᵢ ← clamp(zᵢ Bᵢ^damping, max(lbᵢ, zᵢ - m), min(ubᵢ, zᵢ + m)),   Bᵢ = max(0, -∂F/∂zᵢ) / (λ aᵢ),

with `m = move * (ub - lb)` and `λ > 0` found by bisection on `aᵀz = V`, starting from `[0, 1e9]`. The bisection
stops when `(λ₂ - λ₁)/(λ₁ + λ₂) ≤ bisection_tol` and, as in top88, keeps the design of the last midpoint, so the
constraint holds up to the bisection tolerance. With the defaults, the step is the one of top88.

Requires a linear constraint with `a > 0`. The objective is evaluated once per step (with derivative) and the new
design is not evaluated (`f_new = NaN`, `kkt = NaN`).
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
  dF::V
  z_old::V
end

function init(opt::OC, z, oracle)
  linear_constraint(oracle) === nothing && throw(ArgumentError("OC needs a linear constraint aᵀz ≤ V"))
  OCState(opt.move, similar(z), similar(z))
end

reset!(opt::OC, s::OCState) = (s.move = opt.move; s)
shrink!(opt::OC, s::OCState, factor) = (s.move *= factor; s)

function step!(opt::OC, s::OCState, z, oracle, ctx)
  f = objective!(s.dF, oracle, z)
  a, V = linear_constraint(oracle)
  lb, ub = bounds(oracle)
  copyto!(s.z_old, z)
  zo, dF, η, m = s.z_old, s.dF, opt.damping, s.move

  function update!(λ)
    @. z = clamp(zo * (max(zero(eltype(dF)), -dF) / (λ * a))^η,
                 max(lb, zo - m * (ub - lb)), min(ub, zo + m * (ub - lb)))
    dot(a, z)
  end

  λ1, λ2 = 0.0, 1e9
  n = 0
  while update!(λ2) > V && n < opt.max_bisections   # widen the bracket if the objective is badly scaled
    λ1, λ2 = λ2, 10λ2
    n += 1
  end
  n = 0
  while (λ2 - λ1) / (λ1 + λ2) > opt.bisection_tol && n < opt.max_bisections
    λ = (λ1 + λ2) / 2
    update!(λ) > V ? (λ1 = λ) : (λ2 = λ)   # as in top88, z keeps the update of the last midpoint
    n += 1
  end

  change = maximum(abs, z .- zo)
  StepInfo(true, f, NaN, NaN, change, m, 1)
end
