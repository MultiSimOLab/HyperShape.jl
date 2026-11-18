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
using ..Operators: AbstractOperator, LinearOperator, StateOperator, apply!, solve_linear!, invalidate!, OptContext
using ..Variables: PhysicalFields, update_fields!, design_gradient!, get_designfields, get_optvariables,
                   get_values, get_bounds, set_values!, filter_operators
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
    check_consistency(J1, J2, …; ctx = OptContext(:check), kwargs...) -> ConsistencyReport
    check_consistency((name1 = J1, name2 = J2, …); ctx = OptContext(:check), kwargs...)

Checks, at the current design, that the derivatives the user wrote and the dependencies the user declared are
consistent. The functionals must share the same optimisation variables `z`. Four kinds of checks:

1. **Gradient of each functional** (Taylor test): `J(z + hδ) − J(z) − h⟨∇J, δ⟩` must decrease with order 2 when `h`
   is halved. It covers the partial derivatives of the functional (`dJdu`, `dJdθ`), the adjoint and the `sensitivity` of the state, and the maps x → θ at once.
2. **Design chain of each physical field** (Taylor test of `½∫θ²` through the maps alone, for every `PhysicalFields`
   reached from the functionals): if it passes and the gradient of a functional fails, the error is in the derivatives of the functional or of the state or
   in the adjoint, not in the maps.
3. **Symmetry of each filter**: `cᵀA⁻¹b = bᵀA⁻¹c` for two random vectors, with the user's solve.
4. **Dependencies**: at a perturbed design, value and gradient computed as usual are compared with the ones obtained
   after forcing every state and filter reached through the declared dependencies to assemble and solve again. A
   difference means that one of those operators depends on something it does not declare (for example a filter
   matrix that reads the design with `depends = ()`, or a state listed in `depends` that under-declares its own
   dependencies). The check only changes the design: a `Parameter` read by a model but not declared is not detected
   (it does not change while the check runs). An object that no declared dependency leads to cannot be checked: its
   stale values go unnoticed when the derivatives treat it as fixed too.

Optionally, `tangents = (S => (res, jac), …)` adds a Taylor test of the tangent `jac(u, du, v)` against the residual
`res(u, v)` of the state `S` at its current solution, in a random direction of its free values (the user's forms in
Gridap's convention; for HyperFEM's load-factor forms pass `res(Λ)`, `jac(Λ)`).

A Taylor test passes when two consecutive steps give an observed order of at least `min_rate` (both errors above
rounding), or when every error is at rounding level while J does change along δ (a functional linear in z). It
fails, as *not tested*, when J does not change along δ at all (for example a fully saturated projection): choose
another direction or design. It fails as *inconclusive* when only the largest step gives an error above rounding
(a nearly linear functional): try a larger `h0`. With `h0 = 1e-2` a relative error of about 0.1 % in a gradient can still pass.

Keyword arguments:
- `ctx`: passed to the solve functions (default `OptContext(:check)`).
- `direction`: direction δ in z (default: random, scaled by the width of the bounds, zero within 1e-3 of the width
  from a bound).
- `h0`: first step (default: 90 % of the largest step that keeps z + hδ inside the bounds, at most `1e-2`).
- `steps = 5`: number of steps of each Taylor test (h halved each time).
- `seed = 1`: seed of the random directions and vectors.
- `min_rate = 1.8`: observed order required.
- `rtol = 1e-8`: relative tolerance of the symmetry and dependency checks (increase it with iterative solvers or
  Newton tolerances).

Cost: each forward state is solved at most `steps + 4` times and each adjoint at most 4 times (states shared
by several functionals are solved once per design); each filter is solved once per step for checks 1 and 2, twice
for check 3, and again whenever a gradient is pulled back. Use small meshes for nonlinear problems.

On return, z holds its initial values and the fields and states are brought back to it (one more solve of each
state). The scales of the functionals are the ones they had; a `scale = :initial` not yet fixed stays unfixed (its
functional is checked unscaled), fixed scales are used as they are. If a dependency check fails, the operators may
hold matrices assembled during the check: fix the `depends` and build them again. Errors thrown by the user's forms
or solves propagate (z and the scales are restored first).
"""
check_consistency(Fs::Functional...; kwargs...) =
  check_consistency(NamedTuple{Tuple(Symbol("functional ", i) for i in eachindex(Fs))}(Fs); kwargs...)

function check_consistency(Fs::NamedTuple; ctx = OptContext(:check), direction = nothing, h0 = nothing, steps::Integer = 5,
                           seed::Integer = 1, min_rate::Real = 1.8, rtol::Real = 1e-8, tangents = ())
  isempty(Fs) && throw(ArgumentError("check_consistency: give at least one Functional"))
  all(F -> F isa Functional, values(Fs)) || throw(ArgumentError("check_consistency: every argument must be a Functional"))
  steps >= 2 || throw(ArgumentError("check_consistency: steps must be at least 2"))
  z = _optvariables(first(values(Fs)))
  all(F -> _optvariables(F) === z, values(Fs)) ||
    throw(ArgumentError("check_consistency: the functionals must share the same optimisation variables z"))
  tans = collect(_pairs(tangents))
  all(t -> t isa Pair && t.first isa StateOperator && t.second isa Tuple && length(t.second) == 2, tans) ||
    throw(ArgumentError("check_consistency: tangents must be pairs S => (res, jac) with S a StateOperator"))
  rng = Xoshiro(seed)
  z0 = copy(get_values(z))
  d = _direction(z, direction, rng)
  h = _first_step(z, d, h0)
  hs = [h / 2^(k - 1) for k in 1:steps]
  scales = [F.scale[] for F in values(Fs)]
  ops, θs = _reachable(values(Fs))
  results = CheckResult[]
  completed = false
  try
    for F in values(Fs)                         # an unfixed :initial scale: checked unscaled, restored below
      isnan(F.scale[]) && (F.scale[] = 1.0)
    end
    # tangents first: the states are already solved at the current design
    tangent_results = [_check_tangent(k, length(tans), t.first, t.second..., rng, hs, ctx, min_rate) for (k, t) in enumerate(tans)]
    append!(results, _check_gradients(Fs, z, z0, d, hs, ctx, min_rate))
    for (k, θ) in enumerate(θs)
      append!(results, _check_chains(θ, k, length(θs), z, z0, d, hs, ctx, min_rate))
    end
    filters = _unique(Iterators.flatten(filter_operators(θ) for θ in θs))
    for (k, op) in enumerate(filters)
      push!(results, _check_symmetry(k, op, rng, ctx, rtol))
    end
    append!(results, tangent_results)
    append!(results, _check_dependencies(Fs, ops, θs, z, z0, d, first(hs), ctx, rtol))
    completed = true
  finally
    get_values(z) == z0 || set_values!(z, z0)
    for (F, s) in zip(values(Fs), scales)
      F.scale[] = s
    end
    completed && _bring_back!(ops, θs, ctx)
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

# every operator reachable from the functionals through the declared dependencies, and the physical fields among them
function _reachable(Fs)
  ops, θs, seen = Any[], Any[], Any[]
  stack = Any[]
  for F in Fs
    push!(stack, F.θ)
    if F.S !== nothing
      push!(stack, F.S)
      F.S.adjoint.op isa AbstractOperator && push!(stack, F.S.adjoint.op)   # the adjoint matrix (not with SameMatrix)
    end
  end
  while !isempty(stack)
    o = popfirst!(stack)
    any(y -> y === o, seen) && continue
    push!(seen, o)
    if o isa PhysicalFields
      push!(θs, o)
      append!(stack, filter_operators(o))
    elseif o isa AbstractOperator
      push!(ops, o)
      append!(stack, o.depends)
    end
  end
  ops, θs
end

# fields and states at the restored design
function _bring_back!(ops, θs, ctx)
  foreach(θ -> update_fields!(θ, ctx), θs)
  for o in ops
    o isa StateOperator && apply!(o, ctx)
  end
end

_width(lb, ub, v) = isfinite(ub - lb) ? ub - lb : max(1.0, abs(v))

# random direction scaled by the width of the bounds, zero within 1e-3 of the width from a bound
function _direction(z, d, rng)
  lb, ub = get_bounds(z)
  v = get_values(z)
  if d === nothing
    d = zeros(length(v))
    for i in eachindex(v)
      w = _width(lb[i], ub[i], v[i])
      near = v[i] - lb[i] < 1e-3 * w || ub[i] - v[i] < 1e-3 * w
      d[i] = near ? 0.0 : (2 * rand(rng) - 1) * w
    end
  else
    length(d) == length(v) || throw(DimensionMismatch("check_consistency: the direction has $(length(d)) values, z has $(length(v))"))
    d = Vector{Float64}(d)
    all(isfinite, d) || throw(ArgumentError("check_consistency: the direction must be finite"))
  end
  any(!iszero, d) || throw(ArgumentError("check_consistency: the direction is zero (every value of z is at a bound?)"))
  d
end

# 90 % of the largest step keeping z + hδ inside the bounds, at most 1e-2
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

_set!(z, z0, d, h) = set_values!(z, z0 .+ h .* d)

# errs[k] = |J(z0 + hₖδ) − J(z0) − hₖ⟨g, δ⟩|; vars[k] = |J(z0 + hₖδ) − J(z0)| + |hₖ⟨g, δ⟩| (the size of the change);
# floor: rounding level of J itself
function _taylor_verdict(errs, vars, floor, min_rate)
  any(!isfinite, errs) && return false, "non-finite value at a perturbed design (a state did not converge?)"
  all(v -> v <= floor, vars) &&
    return false, "not tested: J does not change along the direction (choose another direction or design)"
  tol = [1e-9 * vars[k] + floor for k in eachindex(errs)]
  all(k -> errs[k] <= tol[k], eachindex(errs)) &&
    return true, "errors at rounding level (largest $(_fmt(maximum(errs)))): linear along the direction"
  rates = Float64[]
  for k in 1:length(errs)-1
    (errs[k] > tol[k] && errs[k+1] > tol[k+1]) && push!(rates, log2(errs[k] / errs[k+1]))
  end
  ok = any(r -> r >= min_rate, rates)
  # only the largest step above rounding: the order cannot be measured
  (!ok && isempty(rates)) && return false, "inconclusive: only the largest step is above rounding (errors " *
                                           _fmt(first(errs)) * " … " * _fmt(last(errs)) * "); try a larger h0"
  ok, "rates " * (isempty(rates) ? "—" : join((_fmt2(r) for r in rates), " ")) *
      ", errors " * _fmt(first(errs)) * " … " * _fmt(last(errs))
end

_fmt(x) = string(round(x; sigdigits = 2))
_fmt2(x) = isfinite(x) ? string(round(x; digits = 2)) : string(x)
_floor(J0) = 100 * eps() * abs(J0) + floatmin()

# Taylor tests of all functionals together, so that shared states are solved once per step
function _check_gradients(Fs, z, z0, d, hs, ctx, min_rate)
  names, Fv = keys(Fs), collect(values(Fs))
  get_values(z) == z0 || set_values!(z, z0)
  J0 = [functional_value(F, ctx) for F in Fv]
  gd = fill(NaN, length(Fv))
  fail = Vector{Union{Nothing,String}}(nothing, length(Fv))
  for (i, F) in enumerate(Fv)
    if !isfinite(J0[i])
      fail[i] = "the value is not finite at the current design (a state did not converge?)"
      continue
    end
    g = functional_gradient!(zeros(length(z0)), F, ctx)
    all(isfinite, g) ? (gd[i] = dot(g, d)) : (fail[i] = "the gradient is not finite at the current design")
  end
  vals = [Float64[] for _ in Fv]
  if any(isnothing, fail)
    for h in hs
      _set!(z, z0, d, h)
      for (i, F) in enumerate(Fv)
        fail[i] === nothing && push!(vals[i], functional_value(F, ctx))
      end
    end
    set_values!(z, z0)
  end
  map(enumerate(names)) do (i, name)
    label = "gradient of $name"
    fail[i] === nothing || return CheckResult(label, false, fail[i])
    errs = [abs(vals[i][k] - J0[i] - hs[k] * gd[i]) for k in eachindex(hs)]
    vars = [abs(vals[i][k] - J0[i]) + abs(hs[k] * gd[i]) for k in eachindex(hs)]
    ok, msg = _taylor_verdict(errs, vars, _floor(J0[i]), min_rate)
    CheckResult(label, ok, msg)
  end
end

# Taylor tests of ½∫θₙ² through the maps x → θ alone, all physical fields of θ together. Value and derivative use
# the same quadrature, so the test holds whatever its degree.
function _check_chains(θ, k, nθ, z, z0, d, hs, ctx, min_rate)
  get_values(z) == z0 || set_values!(z, z0)
  update_fields!(θ, ctx)
  names = collect(propertynames(θ))
  fs = [getproperty(θ, n) for n in names]
  dΩs = [Measure(get_triangulation(f), 4) for f in fs]
  values_now() = (update_fields!(θ, ctx); [0.5 * sum(∫(f * f)dΩ) for (f, dΩ) in zip(fs, dΩs)])
  J0 = values_now()
  gd = map(eachindex(names)) do i
    f, dΩ = fs[i], dΩs[i]
    forms = NamedTuple{(names[i],)}((δ -> ∫(f * δ)dΩ,))
    dot(design_gradient!(zeros(length(z0)), θ, forms, ctx; key = (:check_consistency, names[i])), d)
  end
  vals = Vector{Float64}[]
  for h in hs
    _set!(z, z0, d, h)
    push!(vals, values_now())
  end
  set_values!(z, z0)
  map(eachindex(names)) do i
    label = "design chain of θ.$(names[i])" * (nθ > 1 ? " (physical fields $k)" : "")
    errs = [abs(vals[s][i] - J0[i] - hs[s] * gd[i]) for s in eachindex(hs)]
    vars = [abs(vals[s][i] - J0[i]) + abs(hs[s] * gd[i]) for s in eachindex(hs)]
    ok, msg = _taylor_verdict(errs, vars, _floor(J0[i]), min_rate)
    CheckResult(label, ok, msg)
  end
end

function _check_symmetry(k, op::LinearOperator, rng, ctx, rtol)
  label = "symmetry of filter $k"
  n = num_free_dofs(last(get_spaces(op.model)))
  b, c = randn(rng, n), randn(rng, n)
  x, y = zeros(n), zeros(n)
  solve_linear!(x, op, b, ctx)
  solve_linear!(y, op, c, ctx)
  cx, by = dot(c, x), dot(b, y)
  ref = max(norm(c) * norm(x), norm(b) * norm(y), floatmin())
  diff = abs(cx - by) / ref
  CheckResult(label, diff <= rtol, "relative difference $(_fmt(diff)) (tolerance $(_fmt(rtol)))")
end

function _check_tangent(k, n, S::StateOperator, res, jac, rng, hs, ctx, min_rate)
  label = n > 1 ? "tangent $k" : "tangent"
  apply!(S, ctx) || return CheckResult(label, false, "the state did not converge at the current design")
  uh = get_state(S)
  U = get_fe_space(uh)
  V = last(get_spaces(S.model))
  u0 = copy(get_free_dof_values(uh))
  w = randn(rng, length(u0)) .* max(norm(u0, Inf), 1.0) ./ sqrt(length(u0))
  R0 = assemble_vector(v -> res(uh, v), V)
  # the tangent applied to the free part of w: jac is linear in du, the Dirichlet values of U cancel
  Jw = assemble_vector(v -> jac(uh, FEFunction(U, w), v), V) - assemble_vector(v -> jac(uh, FEFunction(U, zero(w)), v), V)
  errs, vars = Float64[], Float64[]
  floor = 0.0
  for h in hs
    Rh = assemble_vector(v -> res(FEFunction(U, u0 .+ h .* w), v), V)
    push!(errs, norm(Rh - R0 - h * Jw))
    push!(vars, norm(Rh - R0) + h * norm(Jw))
    floor = max(floor, 100 * eps() * (norm(R0) + norm(Rh)))
  end
  ok, msg = _taylor_verdict(errs, vars, floor + floatmin(), min_rate)
  CheckResult(label, ok, msg)
end

function _values_and_gradients(Fv, n, ctx)
  foreach(invalidate!, Fv)
  vs = [functional_value(F, ctx) for F in Fv]
  gs = [functional_gradient!(zeros(n), F, ctx) for F in Fv]
  vs, gs
end

function _check_dependencies(Fs, ops, θs, z, z0, d, h, ctx, rtol)
  names, Fv = keys(Fs), collect(values(Fs))
  _set!(z, z0, d, h)                          # a design the operators have not seen yet
  v1, g1 = _values_and_gradients(Fv, length(z0), ctx)
  foreach(invalidate!, ops)
  foreach(invalidate!, θs)
  v2, g2 = _values_and_gradients(Fv, length(z0), ctx)
  set_values!(z, z0)
  map(enumerate(names)) do (i, name)
    label = "dependencies of $name"
    (isfinite(v1[i]) && isfinite(v2[i])) || return CheckResult(label, false, "a state did not converge at the perturbed design")
    scale = max(abs(v2[i]), h * norm(g2[i]) * norm(d), floatmin())   # a value near zero is compared with its variation
    dv = abs(v1[i] - v2[i]) / scale
    dg = norm(g1[i] - g2[i]) / max(norm(g2[i]), floatmin())
    CheckResult(label, dv <= rtol && dg <= rtol,
                "relative difference after solving everything again: value $(_fmt(dv)), gradient $(_fmt(dg))")
  end
end

end
