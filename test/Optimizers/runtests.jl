using HyperShape.Optimizers
using HyperShape.Optimizers: init, step!, reset!, shrink!, evaluations
using LinearAlgebra

# Analytical problems with a known solution; no finite elements.

# min ½‖z - t‖²  s.t.  Σ aᵢ zᵢ ≤ V, 0 ≤ z ≤ 1   →   z = clamp(t - μ a, 0, 1)
function quadratic_solution(t, a, V)
  z(μ) = clamp.(t .- μ .* a, 0, 1)
  dot(a, z(0.0)) <= V && return z(0.0)
  lo, hi = 0.0, 1e3
  for _ in 1:200
    μ = (lo + hi) / 2
    dot(a, z(μ)) > V ? (lo = μ) : (hi = μ)
  end
  z(hi)
end

# min Σ cᵢ/zᵢ  s.t.  Σ zᵢ ≤ V, lb ≤ z ≤ 1   →   z = clamp(√(cᵢ/λ), lb, 1)   (compliance-like: ∂F/∂z < 0)
function reciprocal_solution(c, V, lb)
  z(λ) = clamp.(sqrt.(c ./ λ), lb, 1)
  lo, hi = 1e-12, 1e12
  for _ in 1:300
    λ = sqrt(lo * hi)
    sum(z(λ)) > V ? (lo = λ) : (hi = λ)
  end
  z(hi)
end

function run!(opt, z, oracle; iterations, tol = 0.0)
  state = init(opt, z, oracle)
  info = nothing
  for _ in 1:iterations
    info = step!(opt, state, z, oracle, nothing)
    (!info.accepted || info.kkt <= tol) && break
  end
  state, info
end

@testset "Optimizers" begin

  n = 40
  t = [1.5 * sin(0.37i)^2 - 0.2 for i in 1:n]
  c = [0.05 + abs(cos(0.61i)) for i in 1:n]
  a = ones(n)

  @testset "SiMPL: quadratic, volume constraint ($ls)" for ls in (:armijo, :bregman)
    V = 0.3n
    oracle = FunctionOracle(z -> sum(abs2, z .- t) / 2, (dF, z) -> (dF .= z .- t); constraint = (a, V))
    z = fill(0.3, n)
    _, info = run!(SiMPL(linesearch = ls), z, oracle; iterations = 300)
    exact = quadratic_solution(t, a, V)
    F(z) = sum(abs2, z .- t) / 2
    # mirror descent approaches the bounds slowly: loose tolerance on z, tight on the objective
    @test info.accepted
    @test info.kkt <= 1e-3
    @test maximum(abs, z .- exact) < 1e-3
    @test 0 <= F(z) - F(exact) <= 1e-6 * F(exact)
    @test dot(a, z) <= V * (1 + 1e-9)
    @test all(0 .<= z .<= 1)
  end

  @testset "SiMPL: inactive and absent constraint" begin
    exact = clamp.(t, 0, 1)
    for constraint in ((a, 10.0n), nothing)
      oracle = FunctionOracle(z -> sum(abs2, z .- t) / 2, (dF, z) -> (dF .= z .- t); constraint)
      z = fill(0.5, n)
      run!(SiMPL(), z, oracle; iterations = 300, tol = 1e-10)
      @test maximum(abs, z .- exact) < 1e-4
    end
  end

  @testset "SiMPL: monotone decrease and bounds" begin
    V = 0.4n
    oracle = FunctionOracle(z -> sum(c ./ z), (dF, z) -> (dF .= .-c ./ z .^ 2); lb = 1e-3, constraint = (a, V))
    z = fill(0.4, n)
    state = init(SiMPL(), z, oracle)
    values = Float64[]
    for _ in 1:60
      info = step!(SiMPL(), state, z, oracle, nothing)
      push!(values, info.f_new)
      @test info.f_new <= info.f
      @test all(1e-3 .< z .< 1)
    end
    @test issorted(values; rev = true)
  end

  @testset "SiMPL and OC: reciprocal problem" begin
    V, lb = 0.4n, 1e-3
    exact = reciprocal_solution(c, V, lb)
    F(z) = sum(c ./ z)
    oracle() = FunctionOracle(F, (dF, z) -> (dF .= .-c ./ z .^ 2); lb, constraint = (a, V))

    z = fill(0.4, n)
    run!(SiMPL(), z, oracle(); iterations = 500, tol = 1e-9)
    @test maximum(abs, z .- exact) < 1e-3
    @test F(z) <= F(exact) * (1 + 1e-6)

    z = fill(0.4, n)
    o = oracle()
    run!(OC(bisection_tol = 1e-10), z, o; iterations = 200)
    @test maximum(abs, z .- exact) < 1e-3
    @test abs(sum(z) - V) < 1e-6 * V
    @test evaluations(o) == (200, 200)   # one evaluation with derivative per step
  end

  @testset "SiMPL: diagonal metric and vector bounds" begin
    m = [0.5 + (i % 3) for i in 1:n]            # "cell volumes"
    lb, ub = fill(0.1, n), [0.6 + 0.4 * (i % 2) for i in 1:n]
    V = 0.45 * sum(m)
    F(z) = sum(m .* (z .- t) .^ 2) / 2
    oracle = FunctionOracle(F, (dF, z) -> (dF .= m .* (z .- t)); lb, ub, constraint = (m, V), metric = m)
    z = fill(0.45, n)
    _, info = run!(SiMPL(), z, oracle; iterations = 400, tol = 1e-10)
    # exact: z = clamp(t - μ, lb, ub) with Σ mᵢ zᵢ = V
    zμ(μ) = clamp.(t .- μ, lb, ub)
    lo, hi = -10.0, 10.0
    for _ in 1:200
      μ = (lo + hi) / 2
      dot(m, zμ(μ)) > V ? (lo = μ) : (hi = μ)
    end
    @test maximum(abs, z .- zμ(max(hi, 0.0))) < 1e-4
    @test all(lb .<= z .<= ub)
  end

  @testset "OC: move limit, bounds and shrink!/reset!" begin
    V, lb = 0.4n, 1e-3
    oracle = FunctionOracle(z -> sum(c ./ z), (dF, z) -> (dF .= .-c ./ z .^ 2); lb, constraint = (a, V))
    opt = OC(move = 0.2)
    z = fill(0.4, n)
    state = init(opt, z, oracle)
    info = step!(opt, state, z, oracle, nothing)
    @test info.accepted && isnan(info.f_new) && isnan(info.kkt)
    @test info.change <= 0.2 * (1 - lb) + 1e-12
    @test all(lb .<= z .<= 1)
    shrink!(opt, state, 0.5)
    z0 = copy(z)
    info = step!(opt, state, z, oracle, nothing)
    @test info.step ≈ 0.1
    @test maximum(abs, z .- z0) <= 0.1 * (1 - lb) + 1e-12
    reset!(opt, state)
    @test step!(opt, state, z, oracle, nothing).step ≈ 0.2
    @test_throws ArgumentError init(opt, z, FunctionOracle(sum, (dF, z) -> fill!(dF, 1)))
  end

  @testset "SiMPL: shrink! rolls back, reset! forgets" begin
    V = 0.3n
    oracle = FunctionOracle(z -> sum(abs2, z .- t) / 2, (dF, z) -> (dF .= z .- t); constraint = (a, V))
    opt = SiMPL()
    z = fill(0.3, n)
    state = init(opt, z, oracle)
    step!(opt, state, z, oracle, nothing)
    z1 = copy(z)
    info2 = step!(opt, state, z, oracle, nothing)
    # the loop rejects step 2: it restores z and calls shrink!
    copyto!(z, z1)
    shrink!(opt, state, 0.5)
    info2b = step!(opt, state, z, oracle, nothing)
    @test info2b.f ≈ info2.f
    @test info2b.step <= 0.5 * info2.step * (1 + 1e-12)
    # the shrink factor applies to one step only
    info3 = step!(opt, state, z, oracle, nothing)
    @test info3.accepted
    # after reset! the step is the initial one, 1/max|g|, at the current z
    reset!(opt, state)
    g = z .- t
    info = step!(opt, state, z, oracle, nothing)
    @test info.step <= 1 / maximum(abs, g) * (1 + 1e-12)
    @test log2(1 / maximum(abs, g) / info.step) ≈ round(log2(1 / maximum(abs, g) / info.step)) atol = 1e-8
  end

  @testset "SiMPL: failed trial evaluations" begin
    # the objective cannot be evaluated for large steps away from the start
    z0 = fill(0.3, n)
    F(z) = maximum(abs, z .- z0) > 0.05 ? NaN : sum(abs2, z .- t) / 2
    oracle = FunctionOracle(F, (dF, z) -> (dF .= z .- t); constraint = (a, 0.3n))
    z = copy(z0)
    opt = SiMPL()
    state = init(opt, z, oracle)
    info = step!(opt, state, z, oracle, nothing)
    @test info.accepted && isfinite(info.f_new) && info.evaluations > 2
    @test maximum(abs, z .- z0) <= 0.05
    # never acceptable: z is left unchanged
    oracle = FunctionOracle(z -> z == z0 ? 1.0 : NaN, (dF, z) -> (dF .= z .- t))
    z = copy(z0)
    state = init(SiMPL(max_backtracks = 5), z, oracle)
    info = step!(SiMPL(max_backtracks = 5), state, z, oracle, nothing)
    @test !info.accepted && info.evaluations == 7
    @test z == z0
  end

  @testset "Constructor checks" begin
    @test_throws ArgumentError SiMPL(linesearch = :wolfe)
    @test_throws ArgumentError SiMPL(c1 = 2.0)
    @test_throws ArgumentError OC(move = 0.0)
  end

end
