"""
    HyperShape.Checks

Consistency checks of the derivatives and of the declared dependencies: [`check_consistency`](@ref).
"""
module Checks

using LinearAlgebra
using Random
using Gridap
using Gridap.FESpaces
using HyperFEM: get_state, get_spaces
using ..Operators: AbstractOperator, LinearOperator, StateOperator, apply!, is_converged, solve_linear!
using ..Variables: PhysicalFields, FilterStage, update_fields!, design_gradient!, get_designfields, get_optvariables,
                   get_values, get_bounds, set_values!
using ..Functionals: Functional, functional_value, functional_gradient!

export check_consistency

"""
    CheckResult

One line of a [`ConsistencyReport`](@ref): `label`, `passed` and a `message` with the details (observed rates of the
Taylor tests, relative differences).
"""
struct CheckResult
  label::String
  passed::Bool
  message::String
end

"""
    ConsistencyReport

Result of [`check_consistency`](@ref): `results`, a vector of [`CheckResult`](@ref), and `passed`, true when all of
them passed. Displayed as a table.
"""
struct ConsistencyReport
  results::Vector{CheckResult}
  passed::Bool
end
ConsistencyReport(results::Vector{CheckResult}) = ConsistencyReport(results, all(r -> r.passed, results))

function Base.show(io::IO, ::MIME"text/plain", r::ConsistencyReport)
  n = count(x -> x.passed, r.results)
  println(io, "Consistency check: $n of $(length(r.results)) passed")
  w = maximum(x -> length(x.label), r.results; init = 0)
  for x in r.results
    println(io, "  ", x.passed ? "✓" : "✗", " ", rpad(x.label, w), "  ", x.message)
  end
end
Base.show(io::IO, r::ConsistencyReport) =
  print(io, "ConsistencyReport(", count(x -> x.passed, r.results), " of ", length(r.results), " passed)")

"""
    check_consistency(J1, J2, …; ctx = nothing, kwargs...) -> ConsistencyReport
    check_consistency((name1 = J1, name2 = J2, …); ctx = nothing, kwargs...)

Checks, at the current design, that the derivatives the user wrote and the dependencies the user declared are
consistent. The functionals must share the same optimisation variables `z`. Four kinds of checks:

1. **Gradient of each functional** (Taylor test): `J(z + hδ) − J(z) − h⟨∇J, δ⟩` must decrease with order 2 when `h`
   is halved. It covers `DJform`, the adjoint state, the maps x → θ and the scaling at once.
2. **Design chain of each physical field** (Taylor test of `½∫θ²` through the maps): if it passes and the gradient
   of a functional fails, the error is in `DJform` or in the adjoint, not in the maps.
3. **Symmetry of each filter**: `cᵀA⁻¹b = bᵀA⁻¹c` for two random vectors, with the user's solve.
4. **Dependencies of each functional**: at a perturbed design, value and gradient computed as usual are compared
   with the ones obtained after forcing every operator HyperShape can reach (states, filters) to assemble and solve
   again. A difference means that a matrix or a state depends on something it does not declare in `depends`.
   Objects that no declared dependency leads to cannot be reached; their stale values usually make check 1 fail.

Optionally, `tangents = (S => (res, jac), …)` adds a Taylor test of the tangent `jac(u, du, v)` against the residual
`res(u, v)` of the state `S` at its current solution, in a random direction of its free values (these are the
user's forms in Gridap's convention; for HyperFEM's load-factor forms pass `res(Λ)`, `jac(Λ)`).

Keyword arguments:
- `ctx`: passed to the solve functions (default `nothing`).
- `direction`: direction δ in z (default: random, zero where z is at a bound or very close to it).
- `h0`: first step (default: the largest step that keeps z + hδ inside the bounds, at most `1e-2`).
- `steps = 5`: number of steps of each Taylor test (h halved each time).
- `seed = 1`: seed of the random directions and vectors.
- `min_rate = 1.8`: a Taylor test passes when two consecutive steps give an observed order at least `min_rate`, or
  when every error is at the level of rounding (relative `1e-9`, the case of a functional linear in z).
- `rtol = 1e-8`: relative tolerance of the symmetry and dependency checks (increase it with iterative solvers).

Cost: one forward solve per step and per functional (plus the adjoint solves of the gradients), one filter solve per
step and per physical field for check 2, two filter solves per filter for check 3, and a complete re-solve per
functional for check 4. Use small meshes for nonlinear problems.

On return, z holds its initial values (the states are solved again on the next evaluation) and the scale of the
functionals is the one they had: a `scale = :initial` not yet fixed stays unfixed (the checks use unscaled values).
"""
check_consistency(Fs::Functional...; kwargs...) =
  check_consistency(NamedTuple{Tuple(Symbol("functional ", i) for i in eachindex(Fs))}(Fs); kwargs...)

function check_consistency(Fs::NamedTuple; ctx = nothing, direction = nothing, h0 = nothing, steps::Integer = 5,
                           seed::Integer = 1, min_rate::Real = 1.8, rtol::Real = 1e-8, tangents = ())
  isempty(Fs) && throw(ArgumentError("check_consistency: give at least one Functional"))
  all(F -> F isa Functional, values(Fs)) || throw(ArgumentError("check_consistency: every argument must be a Functional"))
  steps >= 2 || throw(ArgumentError("check_consistency: steps must be at least 2"))
  z = _optvariables(first(values(Fs)))
  all(F -> _optvariables(F) === z, values(Fs)) ||
    throw(ArgumentError("check_consistency: the functionals must share the same optimisation variables z"))
  rng = Xoshiro(seed)
  z0 = copy(get_values(z))
  d = _direction(z, direction, rng)
  h = _first_step(z, d, h0)
  hs = [h / 2^(k - 1) for k in 1:steps]
  scales = [F.scale[] for F in values(Fs)]
  results = CheckResult[]
  try
    for F in values(Fs)                         # unscaled checks; scales restored below
      isnan(F.scale[]) && (F.scale[] = 1.0)
    end
    for (name, F) in pairs(Fs)
      push!(results, _check_gradient(String(name), F, z, z0, d, hs, ctx, min_rate))
    end
    for θ in _unique(F.θ for F in values(Fs))
      for n in propertynames(θ)
        push!(results, _check_chain(n, θ, z, z0, d, hs, ctx, min_rate))
      end
    end
    for (k, op) in enumerate(_filters(values(Fs)))
      push!(results, _check_symmetry(k, op, rng, ctx, rtol))
    end
    for (S, (res, jac)) in _pairs(tangents)
      push!(results, _check_tangent(S, res, jac, rng, hs, ctx, min_rate))
    end
    for (name, F) in pairs(Fs)
      push!(results, _check_dependencies(String(name), F, z, z0, d, first(hs), ctx, rtol))
    end
  finally
    get_values(z) == z0 || set_values!(z, z0)
    for (F, s) in zip(values(Fs), scales)
      F.scale[] = s
    end
  end
  ConsistencyReport(results)
end

_optvariables(F::Functional) = get_optvariables(get_designfields(F.θ))

function _unique(itr)
  out = Any[]
  for x in itr
    any(y -> y === x, out) || push!(out, x)
  end
  out
end

_pairs(t::Pair) = (t,)
_pairs(t) = t

# random direction scaled by the width of the bounds, zero within 1e-3 of the width from a bound
function _direction(z, d, rng)
  lb, ub = get_bounds(z)
  v = get_values(z)
  if d === nothing
    d = [(2 * rand(rng) - 1) * (ub[i] - lb[i]) for i in eachindex(v)]
    for i in eachindex(v)
      tol = 1e-3 * (ub[i] - lb[i])
      (v[i] - lb[i] < tol || ub[i] - v[i] < tol) && (d[i] = 0.0)
    end
  else
    length(d) == length(v) || throw(DimensionMismatch("check_consistency: the direction has $(length(d)) values, z has $(length(v))"))
    d = Vector{Float64}(d)
    all(isfinite, d) || throw(ArgumentError("check_consistency: the direction must be finite"))
  end
  any(!iszero, d) || throw(ArgumentError("check_consistency: the direction is zero (every value of z is at a bound?)"))
  d
end

# largest step keeping z + hδ inside the bounds (with a margin), at most 1e-2
function _first_step(z, d, h0)
  lb, ub = get_bounds(z)
  v = get_values(z)
  hmax = Inf
  for i in eachindex(v)
    d[i] > 0 && (hmax = min(hmax, (ub[i] - v[i]) / d[i]))
    d[i] < 0 && (hmax = min(hmax, (lb[i] - v[i]) / d[i]))
  end
  if h0 === nothing
    h = min(1e-2, 0.9 * hmax)
    h > 0 || throw(ArgumentError("check_consistency: the direction leaves the bounds; give another direction"))
    return h
  end
  0 < h0 <= hmax || throw(ArgumentError("check_consistency: h0 = $h0 takes z out of the bounds (largest step $hmax)"))
  Float64(h0)
end

# Taylor test: errs[k] = |J(z0 + hₖδ) − J(z0) − hₖ⟨g, δ⟩|
function _taylor(value_at, J0, gd, hs)
  errs = Float64[]
  refs = Float64[]
  for h in hs
    Jh = value_at(h)
    push!(errs, abs(Jh - J0 - h * gd))
    push!(refs, abs(J0) + abs(Jh) + abs(h * gd))
  end
  errs, refs
end

_rates(errs) = [log2(errs[k] / errs[k+1]) for k in 1:length(errs)-1]

function _taylor_verdict(errs, refs, min_rate)
  any(!isfinite, errs) && return false, "non-finite value at a perturbed design (a state did not converge?)"
  rates = _rates(errs)
  if all(k -> errs[k] <= 1e-9 * max(refs[k], floatmin()), eachindex(errs))
    return true, "errors at rounding level (largest $(_fmt(maximum(errs))))"
  end
  ok = any(r -> r >= min_rate, rates)
  ok, "rates " * join((_fmt2(r) for r in rates), " ") * ", errors " * _fmt(first(errs)) * " … " * _fmt(last(errs))
end

_fmt(x) = string(round(x; sigdigits = 2))
_fmt2(x) = isfinite(x) ? string(round(x; digits = 2)) : string(x)

function _set!(z, z0, d, h)
  set_values!(z, z0 .+ h .* d)
end

function _check_gradient(name, F, z, z0, d, hs, ctx, min_rate)
  label = "gradient of $name"
  get_values(z) == z0 || set_values!(z, z0)
  J0 = functional_value(F, ctx)
  isfinite(J0) || return CheckResult(label, false, "the value is not finite at the current design (a state did not converge?)")
  g = functional_gradient!(zeros(length(z0)), F, ctx)
  all(isfinite, g) || return CheckResult(label, false, "the gradient is not finite at the current design")
  gd = dot(g, d)
  errs, refs = _taylor(h -> (_set!(z, z0, d, h); functional_value(F, ctx)), J0, gd, hs)
  set_values!(z, z0)
  ok, msg = _taylor_verdict(errs, refs, min_rate)
  CheckResult(label, ok, msg)
end

# Taylor test of ½∫θₙ² through the maps x → θ alone
function _check_chain(n, θ, z, z0, d, hs, ctx, min_rate)
  label = "design chain of θ.$n"
  get_values(z) == z0 || set_values!(z, z0)
  update_fields!(θ, ctx)
  f = getproperty(θ, n)
  dΩ = Measure(get_triangulation(f), 4)
  value() = (update_fields!(θ, ctx); 0.5 * sum(∫(f * f)dΩ))
  J0 = value()
  forms = NamedTuple{(n,)}((δ -> ∫(f * δ)dΩ,))
  g = design_gradient!(zeros(length(z0)), θ, forms, ctx; key = (:check_consistency, n))
  gd = dot(g, d)
  errs, refs = _taylor(h -> (_set!(z, z0, d, h); value()), J0, gd, hs)
  set_values!(z, z0)
  ok, msg = _taylor_verdict(errs, refs, min_rate)
  CheckResult(label, ok, msg)
end

# the LinearOperators of the filters in the chains of the physical fields
function _filters(Fs)
  ops = Any[]
  for θ in _unique(F.θ for F in Fs)
    for chain in getfield(θ, :chains), s in chain
      s isa FilterStage && !any(o -> o === s.op, ops) && push!(ops, s.op)
    end
  end
  ops
end

function _check_symmetry(k, op::LinearOperator, rng, ctx, rtol)
  label = "symmetry of filter $k"
  n = _num_rhs(op)
  b, c = randn(rng, n), randn(rng, n)
  x, y = zeros(n), zeros(n)
  solve_linear!(x, op, b, ctx)
  solve_linear!(y, op, c, ctx)
  cx, by = dot(c, x), dot(b, y)
  ref = max(norm(c) * norm(x), norm(b) * norm(y), floatmin())
  diff = abs(cx - by) / ref
  CheckResult(label, diff <= rtol, "relative difference $(_fmt(diff)) (tolerance $(_fmt(rtol)))")
end

_num_rhs(op::LinearOperator) = num_free_dofs(last(get_spaces(op.model)))

function _check_tangent(S::StateOperator, res, jac, rng, hs, ctx, min_rate)
  label = "tangent of a state"
  apply!(S, ctx) || return CheckResult(label, false, "the state did not converge at the current design")
  uh = get_state(S)
  U = get_fe_space(uh)
  V = last(get_spaces(S.model))
  u0 = copy(get_free_dof_values(uh))
  w = randn(rng, length(u0)) .* max(norm(u0, Inf), 1.0) ./ sqrt(length(u0))
  wh = FEFunction(U, w)
  zh = FEFunction(U, zero(w))
  R0 = assemble_vector(v -> res(uh, v), V)
  Jw = assemble_vector(v -> jac(uh, wh, v), V) - assemble_vector(v -> jac(uh, zh, v), V)   # the free part of w only
  errs, refs = Float64[], Float64[]
  for h in hs
    Rh = assemble_vector(v -> res(FEFunction(U, u0 .+ h .* w), v), V)
    push!(errs, norm(Rh - R0 - h * Jw))
    push!(refs, norm(R0) + norm(Rh) + h * norm(Jw))
  end
  ok, msg = _taylor_verdict(errs, refs, min_rate)
  CheckResult(label, ok, msg)
end

# every operator reachable from F through the declared dependencies, and the physical fields among them
function _reachable(F)
  ops, θs = Any[], Any[]
  stack = Any[F.θ]
  F.S === nothing || push!(stack, F.S)
  F.A === nothing || push!(stack, F.A)
  seen = Any[]
  while !isempty(stack)
    o = pop!(stack)
    any(y -> y === o, seen) && continue
    push!(seen, o)
    if o isa PhysicalFields
      push!(θs, o)
      for chain in getfield(o, :chains), s in chain
        s isa FilterStage && push!(stack, s.op)
      end
    elseif o isa AbstractOperator
      push!(ops, o)
      append!(stack, o.depends)
    end
  end
  ops, θs
end

function _invalidate!(ops, θs)
  for o in ops
    o isa StateOperator && (o.solved = false)
    o isa LinearOperator && (o.force_stale = true)       # the next solve is stale: the user re-assembles
  end
  for θ in θs
    getfield(θ, :seen)[] = -1                             # the chains are recomputed, the version grows
  end
end

function _value_and_gradient(F, n, ctx)
  F.cache[1] = -1.0
  v = functional_value(F, ctx)
  g = functional_gradient!(zeros(n), F, ctx)
  v, g
end

function _check_dependencies(name, F, z, z0, d, h, ctx, rtol)
  label = "dependencies of $name"
  _set!(z, z0, d, h)                          # a design the operators have not seen yet
  v1, g1 = _value_and_gradient(F, length(z0), ctx)
  ops, θs = _reachable(F)
  _invalidate!(ops, θs)
  v2, g2 = _value_and_gradient(F, length(z0), ctx)
  set_values!(z, z0)
  (isfinite(v1) && isfinite(v2)) || return CheckResult(label, false, "a state did not converge at the perturbed design")
  dv = abs(v1 - v2) / max(abs(v2), floatmin())
  dg = norm(g1 - g2) / max(norm(g2), floatmin())
  ok = dv <= rtol && dg <= rtol
  msg = "relative difference after solving everything again: value $(_fmt(dv)), gradient $(_fmt(dg))"
  CheckResult(label, ok, msg)
end

end
