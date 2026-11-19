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

# min Σ cᵢ/zᵢ  s.t.  g(z) = Σ zᵢ²/W - 1 ≤ 0, lb ≤ z ≤ 1   →   z = clamp((cᵢ W / (2λ))^(1/3), lb, 1)
function nonlinear_solution(c, W, lb)
  z(λ) = clamp.(cbrt.(c .* W ./ (2λ)), lb, 1)
  lo, hi = 1e-12, 1e12
  for _ in 1:300
    λ = sqrt(lo * hi)
    sum(abs2, z(λ)) > W ? (lo = λ) : (hi = λ)
  end
  z(hi)
end

step_allocations(opt, state, z, oracle) = @allocated step!(opt, state, z, oracle, nothing)

function run!(opt, z, oracle; iterations, tol = 0.0)
  state = init(opt, z, oracle)
  info = nothing
  for _ in 1:iterations
    info = step!(opt, state, z, oracle, nothing)
    (!info.accepted || info.kkt <= tol) && break
  end
  state, info
end

# a vector that refuses scalar indexing outside broadcasting (as a GPU array)
struct NoScalarVector <: AbstractVector{Float64}; v::Vector{Float64}; end
const ALLOW_SCALAR = Ref(false)
Base.size(x::NoScalarVector) = size(x.v)
Base.getindex(x::NoScalarVector, i::Int) = (ALLOW_SCALAR[] || error("scalar getindex"); x.v[i])
Base.setindex!(x::NoScalarVector, a, i::Int) = (ALLOW_SCALAR[] || error("scalar setindex!"); x.v[i] = a)
Base.similar(x::NoScalarVector, ::Type{Float64}, d::Dims{1}) = NoScalarVector(similar(x.v, d))
Base.BroadcastStyle(::Type{NoScalarVector}) = Broadcast.ArrayStyle{NoScalarVector}()
Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{NoScalarVector}}, ::Type{T}) where T = NoScalarVector(Vector{T}(undef, length(bc)))
function Base.copyto!(d::NoScalarVector, bc::Broadcast.Broadcasted{Nothing})
  ALLOW_SCALAR[] = true; try copyto!(d.v, bc) finally ALLOW_SCALAR[] = false end; d
end
Base.copyto!(d::NoScalarVector, s::NoScalarVector) = (copyto!(d.v, s.v); d)
Base.copyto!(d::Vector{Float64}, s::NoScalarVector) = copyto!(d, s.v)
Base.copyto!(d::NoScalarVector, s::AbstractVector) = (copyto!(d.v, s); d)
Base.mapreduce(f, op, x::NoScalarVector; kw...) = mapreduce(f, op, x.v; kw...)
Base.mapreduce(f, op, x::NoScalarVector, y::NoScalarVector; kw...) = mapreduce(f, op, x.v, y.v; kw...)
LinearAlgebra.dot(x::NoScalarVector, y::NoScalarVector) = dot(x.v, y.v)
Base.sum(x::NoScalarVector) = sum(x.v)
Base.fill!(x::NoScalarVector, a) = (fill!(x.v, a); x)
# reductions of lazy broadcasts, as GPU arrays do them (materialised here)
Base.mapreduce(f, op, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{NoScalarVector}}; kw...) = mapreduce(f, op, copy(bc).v; kw...)
Base.sum(f, x::NoScalarVector) = sum(f, x.v)
Base.copy(x::NoScalarVector) = NoScalarVector(copy(x.v))
Base.maximum(x::NoScalarVector) = maximum(x.v)
Base.maximum(f, x::NoScalarVector) = maximum(f, x.v)
Base.minimum(x::NoScalarVector) = minimum(x.v)
Base.any(f::Function, x::NoScalarVector) = any(f, x.v)
Base.all(f::Function, x::NoScalarVector) = all(f, x.v)

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
      @test info.accepted
      @test info.f_new <= info.f * (1 + 1e-13)   # monotone up to rounding
      @test all(1e-3 .< z .< 1)
    end
    @test values[end] < values[1]
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
    @test violation(info) <= 1e-2 * V
    @test all(lb .<= z .<= 1)
    shrink!(opt, state, 0.5)
    z0 = copy(z)
    info = step!(opt, state, z, oracle, nothing)
    @test info.step ≈ 0.1
    @test maximum(abs, z .- z0) <= 0.1 * (1 - lb) + 1e-12
    reset!(opt, state)
    @test step!(opt, state, z, oracle, nothing).step ≈ 0.2
    @test_throws ArgumentError init(opt, z, FunctionOracle(sum, (dF, z) -> fill!(dF, 1)))
    @test_throws ArgumentError init(opt, z, FunctionOracle(sum, (dF, z) -> fill!(dF, 1); constraint = (zeros(n), V)))
  end

  @testset "OC: infeasible start is reported" begin
    V, lb = 0.4n, 1e-3
    oracle = FunctionOracle(z -> sum(c ./ z), (dF, z) -> (dF .= .-c ./ z .^ 2); lb, constraint = (a, V))
    z = fill(0.9, n)
    opt = OC()
    info = step!(opt, init(opt, z, oracle), z, oracle, nothing)
    @test info.accepted
    @test violation(info) ≈ sum(z) - V
    @test violation(info) > 0.1 * V
  end

  quadratic(V = 0.3n) = FunctionOracle(z -> sum(abs2, z .- t) / 2, (dF, z) -> (dF .= z .- t); constraint = (a, V))

  @testset "SiMPL: shrink! rolls back and accumulates" begin
    oracle = quadratic()
    opt = SiMPL()
    z = fill(0.3, n)
    state = init(opt, z, oracle)
    step!(opt, state, z, oracle, nothing)
    z1 = copy(z)
    ψ1 = copy(state.ψ)
    info2 = step!(opt, state, z, oracle, nothing)
    @test info2.evaluations == 1                      # first trial accepted
    nder = evaluations(oracle)[2]

    # the loop rejects step 2: it restores z and calls shrink!
    copyto!(z, z1)
    shrink!(opt, state, 0.5)
    @test state.ψ == ψ1
    a1 = step!(opt, state, z, oracle, nothing)
    @test a1.f == info2.f
    @test a1.step ≈ 0.5 * info2.step
    # rejected again: the reduction accumulates
    copyto!(z, z1)
    shrink!(opt, state, 0.5)
    @test state.ψ == ψ1
    a2 = step!(opt, state, z, oracle, nothing)
    @test a2.step ≈ 0.25 * info2.step
    @test evaluations(oracle)[2] == nder              # the retries reuse the derivative
    # the step stands: the reduction is dropped
    step!(opt, state, z, oracle, nothing)
    @test state.scale == 1
  end

  @testset "SiMPL: shrink! without a step to roll back" begin
    opt = SiMPL()
    g0 = fill(0.3, n) .- t
    # before the first step
    oracle = quadratic()
    z = fill(0.3, n)
    state = init(opt, z, oracle)
    shrink!(opt, state, 0.5)
    @test step!(opt, state, z, oracle, nothing).step ≈ 0.5 / maximum(abs, g0)
    # after reset!: the history stays forgotten
    for _ in 1:3
      step!(opt, state, z, oracle, nothing)
    end
    reset!(opt, state)
    shrink!(opt, state, 0.5)
    @test state.k == 0
    g = z .- t
    @test step!(opt, state, z, oracle, nothing).step ≈ 0.5 / maximum(abs, g)
    # reset! alone: initial step at the current z
    reset!(opt, state)
    g = z .- t
    @test step!(opt, state, z, oracle, nothing).step ≈ 1 / maximum(abs, g)
  end

  @testset "SiMPL: shrink! keeps a saturated latent variable" begin
    oracle = quadratic()
    opt = SiMPL()
    z = fill(0.3, n)
    state = init(opt, z, oracle)
    for _ in 1:150
      step!(opt, state, z, oracle, nothing)
    end
    zb, ψb = copy(z), copy(state.ψ)
    @test maximum(abs, ψb) > 50
    step!(opt, state, z, oracle, nothing)
    copyto!(z, zb)
    shrink!(opt, state, 0.5)
    @test state.ψ == ψb
    @test state.z == zb
  end

  @testset "SiMPL: behaviour at numerical convergence ($ls)" for ls in (:armijo, :bregman)
    V, lb = 0.4n, 1e-3
    oracle = FunctionOracle(z -> sum(c ./ z), (dF, z) -> (dF .= .-c ./ z .^ 2); lb, constraint = (a, V))
    opt = SiMPL(linesearch = ls)
    z = fill(0.3, n)
    state = init(opt, z, oracle)
    infos = [step!(opt, state, z, oracle, nothing) for _ in 1:200]
    @test all(i -> i.accepted, infos)
    @test all(i -> isfinite(i.kkt), infos)
    @test infos[end].kkt < 1e-6                              # function values resolve the design up to ~sqrt(eps)
    @test maximum(i -> i.evaluations, infos) < 31            # no exhausted line search (31 trials)
    @test maximum(violation, infos) <= 1e-10 * V
    @test maximum(abs, z .- reciprocal_solution(c, V, lb)) < 1e-6
  end

  @testset "SiMPL: one forward evaluation per trial" begin
    oracle = quadratic()
    opt = SiMPL()
    z = fill(0.3, n)
    state = init(opt, z, oracle)
    infos = [step!(opt, state, z, oracle, nothing) for _ in 1:50]
    nvalue, nderivative = evaluations(oracle)
    @test nvalue == sum(i -> i.evaluations, infos)
    @test nvalue == 1 + sum(i -> i.evaluations, infos[2:end]) + (infos[1].evaluations - 1)
    @test infos[1].evaluations >= 2                  # value at the start + first trial
    @test nderivative == 50
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
    oracle = FunctionOracle(z -> maximum(abs, z .- z0) < 1e-12 ? 1.0 : NaN, (dF, z) -> (dF .= z .- t))
    z = copy(z0)
    opt = SiMPL(max_backtracks = 5)
    state = init(opt, z, oracle)
    info = step!(opt, state, z, oracle, nothing)
    @test !info.accepted && info.evaluations == 7 && isfinite(info.kkt)
    @test z == z0
    # the objective fails at the starting design
    oracle = FunctionOracle(z -> NaN, (dF, z) -> (dF .= z .- t))
    info = step!(opt, init(opt, z, oracle), z, oracle, nothing)
    @test !info.accepted && info.evaluations == 1
    @test evaluations(oracle) == (1, 0)
  end

  @testset "No allocations per step" begin
    oracle = let t = t
      FunctionOracle(z -> sum(i -> abs2(z[i] - t[i]), eachindex(z)) / 2, (dF, z) -> (dF .= z .- t); constraint = (a, 0.3n))
    end
    for opt in (SiMPL(), SiMPL(linesearch = :bregman))
      z = fill(0.3, n)
      state = init(opt, z, oracle)
      step!(opt, state, z, oracle, nothing); step!(opt, state, z, oracle, nothing)
      @test step_allocations(opt, state, z, oracle) == 0
    end
    oracle = let c = c
      FunctionOracle(z -> sum(i -> c[i] / z[i], eachindex(z)), (dF, z) -> (dF .= .-c ./ z .^ 2); lb = 1e-3, constraint = (a, 0.4n))
    end
    opt = OC()
    z = fill(0.4, n)
    state = init(opt, z, oracle)
    step!(opt, state, z, oracle, nothing)
    @test step_allocations(opt, state, z, oracle) == 0
  end

  @testset "SiMPL: growth of the step is capped" begin
    for (ls, growth) in ((:armijo, 10.0), (:bregman, 10.0), (:bregman, 2.0))
      oracle = quadratic()
      opt = SiMPL(linesearch = ls, max_growth = growth)
      z = fill(0.3, n)
      state = init(opt, z, oracle)
      steps = [step!(opt, state, z, oracle, nothing).step for _ in 1:200]
      @test all(steps[2:end] .<= growth .* steps[1:end-1] .* (1 + 1e-12))
    end
  end

  @testset "Vectors without scalar indexing (GPU arrays): OC and SiMPL" begin
    W = 0.1n
    lin() = FunctionOracle(z -> 0.5 * sum(abs2, z.v .- t), (dF, z) -> (dF .= z .- NoScalarVector(t)); constraint = (NoScalarVector(copy(a)), 0.3n))
    nonlin() = FunctionOracle(z -> 0.5 * sum(abs2, z.v .- t), (dF, z) -> (dF .= z .- NoScalarVector(t));
                              constraint = (z -> sum(abs2, z) / W - 1, (dG, z) -> (dG .= 2 .* z ./ W)),
                              metric = NoScalarVector([1.0 + 0.1 * cos(i) for i in 1:n]))
    for opt in (OC(), SiMPL()), (mk, g) in ((lin, z -> sum(z) / (0.3n) - 1), (nonlin, z -> sum(abs2, z) / W - 1))
      z = NoScalarVector(fill(0.3, n))
      o = mk()
      s = init(opt, z, o)
      infos = [step!(opt, s, z, o, nothing) for _ in 1:30]
      @test all(i -> i.accepted, infos)
      @test g(z) <= 1e-3 && infos[end].f < infos[1].f
    end
  end

  @testset "Constructor checks and names" begin
    @test_throws ArgumentError SiMPL(max_growth = 0.5)
    @test_throws ArgumentError SiMPL(linesearch = :wolfe)
    @test_throws ArgumentError SiMPL(c1 = 2.0)
    @test_throws ArgumentError OC(move = 0.0)
    z = fill(0.3, n)
    @test_throws ArgumentError init(SiMPL(), z, quadratic(0.3n) |> o -> FunctionOracle(o.f, o.df!; constraint = (-a, 1.0)))
    @test !(:inner in names(HyperShape.Optimizers))   # Gridap exports another `inner`
  end

  @testset "Nonlinear constraint: OC and SiMPL (feasible and infeasible start)" begin
    lb = 1e-3
    W = 0.1n
    F(z) = sum(c ./ z)
    g(z) = sum(abs2, z) / W - 1
    mk() = FunctionOracle(F, (dF, z) -> (dF .= -c ./ z .^ 2); lb, ub = 1.0,
                          constraint = (g, (dG, z) -> (dG .= 2 .* z ./ W)))
    exact = nonlinear_solution(c, W, lb)
    @test num_constraints(FunctionOracle(F, (dF, z) -> dF)) == 0 && num_constraints(mk()) == 1
    @test linear_constraint(mk(), 1) === nothing
    @test_throws ArgumentError FunctionOracle(F, (dF, z) -> dF; constraint = (1.0, 2.0))
    # OC: linearised constraint corrected to |g| ≤ constraint_tol
    z = fill(0.3, n)
    s, info = run!(OC(), z, mk(); iterations = 300)
    @test abs(g(z)) <= 1e-4 && violation(info) <= 1e-4
    @test maximum(abs, z .- exact) < 1e-2
    # SiMPL: every accepted iterate feasible, -tol ≤ g ≤ 0
    for z0 in (0.3, 0.6)   # 0.6: infeasible start, restoration steps first
      z = fill(z0, n)
      o = mk()
      st = init(SiMPL(), z, o)
      gs = Float64[]
      for _ in 1:300
        info = step!(SiMPL(), st, z, o, nothing)
        info.accepted || break
        push!(gs, g(z))
      end
      @test length(gs) == 300
      @test -1e-6 <= gs[end] <= 0
      @test all(<=(1e-12), gs[20:end])
      @test maximum(abs, z .- exact) < 1e-3
      @test 0 <= F(z) - F(exact) <= 1e-5 * F(exact)
    end
    # the shift is forgotten by reset!
    st = init(SiMPL(), fill(0.3, n), mk())
    st.shift = 0.5
    reset!(SiMPL(), st)
    @test st.shift == 0
    so = init(OC(), fill(0.3, n), mk())
    so.shift = 0.5
    reset!(OC(), so)
    @test so.shift == 0
    @test_throws ArgumentError OC(constraint_tol = 0)
    @test_throws ArgumentError SiMPL(max_corrections = -1)
    @test constraint_tol(OC()) == 1e-4 && constraint_tol(SiMPL()) == 1e-6
    # a rejected step (shrink!) scales the shift with the square of the factor
    st = init(SiMPL(), fill(0.3, n), mk())
    st.shift = 0.5
    shrink!(SiMPL(), st, 0.5)
    @test st.shift == 0.125
  end

  @testset "OC: adaptive move limits per entry" begin
    o = FunctionOracle(z -> sum(x -> abs2(x - 0.9), z) / 2, (d, z) -> (d .= z .- 0.9); constraint = (ones(4), 1.6))
    opt = OC()
    z = fill(0.4, 4)
    s = init(opt, z, o)
    @test s.m == fill(0.2, 4)
    # last step +, +, -, 0 after a previous step +, -, -, +: the sign test
    s.k = 1
    s.dprev .= [1.0, 1.0, -1.0, 1.0]
    s.z_old .= z .- [0.1, -0.1, -0.1, 0.0]
    s.m .= [0.1, 0.1, 0.1, 0.1]
    m = HyperShape.Optimizers._move_limits!(opt, s, z)
    @test s.m ≈ [0.12, 0.07, 0.12, 0.1]
    @test m == s.m                           # no shrink! factor
    s.m .= [0.19, 0.0011, 0.1, 0.1]
    s.dprev .= [1.0, -1.0, 0.0, 0.0]
    s.z_old .= z .- [0.1, 0.1, 0.0, 0.0]
    HyperShape.Optimizers._move_limits!(opt, s, z)
    @test s.m ≈ [0.2, 0.001, 0.1, 0.1]       # ceiling and floor
    shrink!(opt, s, 0.5)
    s.k = 0                                  # no sign test: only the scaling
    m = HyperShape.Optimizers._move_limits!(opt, s, z)
    @test m ≈ [0.1, 0.0005, 0.05, 0.05] && s.m ≈ [0.2, 0.001, 0.1, 0.1]
    reset!(opt, s)
    @test s.m == fill(0.2, 4) && s.k == 0 && all(iszero, s.dprev)
    # no allocations with adaptive limits; adapt = nothing keeps the scalar limit of top88
    z = fill(0.4, 4)
    s = init(opt, z, o)
    step!(opt, s, z, o, nothing); step!(opt, s, z, o, nothing)
    @test step_allocations(opt, s, z, o) == 0
    o2 = OC(adapt = nothing)
    z2 = fill(0.4, 4)
    s2 = init(o2, z2, o)
    @test step!(o2, s2, z2, o, nothing).step == 0.2
    @test set_move!(o2, 0.1).move == 0.1
    @test set_move!(o2, 5e-4).move == 5e-4                 # no floor without adapt
    @test_throws ArgumentError set_move!(OC(), 5e-4)       # below move_min
    # a rejected step: the retry only rescales, and the next step tests the sign against the last accepted step
    t = [0.9, 0.7, 0.5, 0.3]                                # optimum t .- 0.2: the entries move both ways
    o3 = FunctionOracle(z -> sum(i -> abs2(z[i] - t[i]), eachindex(z)) / 2, (d, z) -> (d .= z .- t);
                        constraint = (ones(4), 1.6))
    opt = OC()
    z = fill(0.4, 4)
    s = init(opt, z, o3)
    for _ in 1:3; step!(opt, s, z, o3, nothing); end
    d1, m1 = copy(s.dprev), copy(s.m)
    @test any(!iszero, d1)
    copyto!(z, s.z_old)                                    # the loop rolls the step back
    shrink!(opt, s, 0.5)
    info = step!(opt, s, z, o3, nothing)
    @test s.m == m1 && s.dprev == d1 && s.meff ≈ 0.5 .* m1 && info.step ≈ sum(m1) / 8
    zr, zr_old = copy(z), copy(s.z_old)
    step!(opt, s, z, o3, nothing)
    @test s.dprev == zr .- zr_old && s.meff == s.m           # the reduction is dropped once the step stood
    # set_move! without reset!: the limits are capped from the next step
    set_move!(opt, 0.05)
    @test step!(opt, s, z, o3, nothing).step <= 0.05 + 1e-12
    # memory for a checkpoint: a state restored from it takes the same next step
    v = memory(opt, s, z)
    s2 = init(opt, copy(z), o3)
    restore_memory!(opt, s2, z, v)
    za, zb = copy(z), copy(z)
    step!(opt, s, za, o3, nothing); step!(opt, s2, zb, o3, nothing)
    @test za == zb && s.m == s2.m && s.dprev == s2.dprev
    @test_throws ArgumentError restore_memory!(opt, s2, z, v[1:end-1])
    # no allocations with adaptive limits and a nonlinear constraint
    W = 1.0
    onl = FunctionOracle(z -> sum(x -> abs2(x - 0.9), z) / 2, (d, z) -> (d .= z .- 0.9);
                         constraint = (z -> sum(abs2, z) / W - 1, (d, z) -> (d .= 2 .* z ./ W)))
    opt = OC()
    z = fill(0.4, 4)
    s = init(opt, z, onl)
    step!(opt, s, z, onl, nothing); step!(opt, s, z, onl, nothing)
    @test step_allocations(opt, s, z, onl) == 0
  end

  @testset "Nonlinear constraint: review cases" begin
    lb = 1e-3
    F(z) = sum(c ./ z)
    dF!(dF, z) = (dF .= -c ./ z .^ 2)
    # restoration where the linearised target lies below aᵀlb (g concave): no error, feasibility reached
    gs(z) = sum(sqrt, z) / n - 0.3
    o = FunctionOracle(F, dF!; lb, ub = 1.0, constraint = (gs, (dG, z) -> (dG .= 0.5 ./ (n .* sqrt.(z)))))
    z = fill(0.9, n)
    st = init(SiMPL(), z, o)
    for _ in 1:30
      step!(SiMPL(), st, z, o, nothing).accepted || break
    end
    @test gs(z) <= 0
    # corrections exhausted: an infeasible trial is not accepted (every accepted iterate stays feasible)
    gc(z) = sum(z .^ 3) / n - 0.2
    o = FunctionOracle(F, dF!; lb, ub = 1.0, constraint = (gc, (dG, z) -> (dG .= 3 .* z .^ 2 ./ n)))
    for mc in (0, 1)
      z = fill(0.3, n)
      st = init(SiMPL(max_corrections = mc), z, o)
      for _ in 1:40
        info = step!(SiMPL(max_corrections = mc), st, z, o, nothing)
        info.accepted || break
        @test gc(z) <= 0
      end
    end
    # entries the constraint does not depend on (zero derivative): OC moves them by the move limit, no error
    gh(z) = sum(z[1:n÷2]) / n - 0.2
    o = FunctionOracle(F, dF!; lb, ub = 1.0, constraint = (gh, (dG, z) -> (dG .= 0; dG[1:n÷2] .= 1 / n; dG)))
    z = fill(0.3, n)
    info = step!(OC(), init(OC(), z, o), z, o, nothing)
    @test info.accepted && all(>(0.49), z[n÷2+1:end]) && gh(z) <= 1e-4   # +move limit, 0.2 (ub - lb)
  end


  @testset "Stationarity measure" begin
    n = 40
    t = [1.5 * sin(0.37i)^2 - 0.2 for i in 1:n]
    c = [0.05 + abs(cos(0.61i)) for i in 1:n]
    a = ones(n)
    # quadratic: g = z - t (identity metric); at the solution S = 0, elsewhere S = ‖z − P(z − g)‖ with P computed here
    V = 0.3n
    o = FunctionOracle(z -> sum(abs2, z .- t) / 2, (dF, z) -> (dF .= z .- t); constraint = (a, V))
    exact = quadratic_solution(t, a, V)
    w = zeros(n)
    @test stationarity!(w, o, exact, exact .- t, a, a, V) < 1e-10
    z = fill(0.3, n)
    y = z .- (z .- t)                                        # = t
    @test stationarity!(w, o, z, z .- t, a, a, V) ≈ norm(z .- quadratic_solution(y, a, V)) rtol = 1e-9
    # without constraint: plain clamp; with an inactive constraint the same
    @test stationarity!(w, o, z, z .- t, nothing, nothing, NaN) ≈ norm(z .- clamp.(t, 0, 1)) rtol = 1e-12
    @test stationarity!(w, o, z, z .- t, a, a, 10n) ≈ norm(z .- clamp.(t, 0, 1)) rtol = 1e-12
    # diagonal metric: the norm is ‖⋅‖_M and g = M⁻¹∂F
    m = [0.5 + 0.01i for i in 1:n]
    om = FunctionOracle(z -> 0.0, (dF, z) -> dF; metric = m)
    g = 0.1 .* ones(n)
    @test stationarity!(w, om, z, g, nothing, nothing, NaN) ≈ sqrt(sum(m .* (z .- clamp.(z .- g, 0, 1)) .^ 2)) rtol = 1e-12
    # non-uniform metric, vector bounds and an active constraint: aᵀP = V and the variational inequality
    #   (y − P)ᵀM(v − P) ≤ 0 for feasible v, y = z − g
    lbv, ubv = [0.05 + 0.002i for i in 1:n], [0.9 - 0.001i for i in 1:n]
    av = [0.5 + 0.03i for i in 1:n]
    Vv = 0.35 * dot(av, ubv)
    ov = FunctionOracle(z -> 0.0, (dF, z) -> dF; lb = lbv, ub = ubv, constraint = (av, Vv), metric = m)
    zv = clamp.(fill(0.4, n), lbv, ubv)
    gv = [0.8 * cos(0.3i) - 0.6 for i in 1:n]
    Sv = stationarity!(w, ov, zv, gv, av ./ m, av, Vv)
    P = zv .+ w
    @test dot(av, P) ≈ Vv rtol = 1e-10
    @test all(lbv .- 1e-14 .<= P .<= ubv .+ 1e-14)              # P = z + (P − z): rounding
    y = zv .- gv
    for k in 1:20
      v = clamp.(P .+ 0.05 .* [sin(1.3 * i * k) for i in 1:n], lbv, ubv)
      dot(av, v) <= Vv || continue
      @test sum(m .* (y .- P) .* (v .- P)) <= 1e-8
    end
    @test Sv ≈ sqrt(sum(m .* w .^ 2)) rtol = 1e-12
    ga = av ./ m
    @test (@allocated stationarity!(w, ov, zv, gv, ga, av, Vv)) == 0
    # V below the minimum of aᵀz over the bounds, or not finite: NaN
    @test isnan(stationarity!(w, ov, zv, gv, ga, av, 0.5 * dot(av, lbv)))
    @test isnan(stationarity!(w, ov, zv, gv, ga, av, NaN))
    # computed by SiMPL and OC; it goes to zero at convergence
    for opt in (SiMPL(), OC(bisection_tol = 1e-12))
      oz = FunctionOracle(z -> sum(c ./ z), (dF, z) -> (dF .= .-c ./ z .^ 2); lb = 1e-3, constraint = (a, 0.4n))
      zz = fill(0.4, n)
      st = init(opt, zz, oz)
      s1 = step!(opt, st, zz, oz, nothing).stationarity
      local info
      for _ in 1:300
        info = step!(opt, st, zz, oz, nothing)
      end
      @test isfinite(s1) && s1 > 0
      @test info.stationarity < 1e-4 * s1
    end
  end

end
