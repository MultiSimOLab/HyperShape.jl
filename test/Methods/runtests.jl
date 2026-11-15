using Gridap
using Gridap.FESpaces
using HyperFEM
using HyperShape
using HyperShape.Variables: get_values, set_values!, set_beta!
using HyperShape.Operators: num_solves
using HyperShape.Optimizers: FunctionOracle, OC, SiMPL, init, step!, shrink!
using HyperShape.Methods: FunctionalOracle, _setup_constraint!
using LinearAlgebra
using Test

# heat compliance (self-adjoint) and the volume of the unprojected design density
function compliance_problem(; design = :P0, n = 8, dim = 2, init = 0.4, fraction = 0.4, solve_flag = nothing)
  p = state_problem(; design, n, dim, init)
  (; z, x, θ, S, f, dk, dΩ) = p
  Sc = solve_flag === nothing ? S :
       StateOperator(S.model; depends = θ, solve = (m, ctx) -> (push!(p.ctxs, (:S, ctx)); (solve!(m; Assembly = true), solve_flag())))
  C = Functional((u, θ) -> ∫(f * u)dΩ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), Sc, Sc, θ; scale = :initial)
  θv = PhysicalFields(x; ρ = Identity())
  vol = sum(∫(1.0)dΩ)
  Vol = Functional(θ -> sum(∫(θ.ρ)dΩ) - fraction * vol, θ -> (δ -> ∫(δ)dΩ), θv; scale = fraction * vol)
  (; p..., C, Vol, θv, S = Sc, problem = Optimization(z; objective = C, linear_constraints = (Vol,)))
end

nonincreasing(v) = all(i -> v[i+1] <= v[i] + 1e-12 * abs(v[i]), 1:length(v)-1)

@testset "Methods" begin

  @testset "Oracle: volume vector, linearity, metric ($(design))" for design in (:P0, :P1)
    q = compliance_problem(; design)
    m = collect(assemble_vector(v -> ∫(v)q.dΩ, q.Vρ))
    o = FunctionalOracle(q.problem, LumpedMass(q.dΩ))
    @test o.metric ≈ m
    design === :P0 && @test all(≈(1 / 64), o.metric)
    _setup_constraint!(o, q.Vol, q.problem.ctx)
    @test o.a ≈ m ./ (0.4 * sum(m))
    @test o.V ≈ 1.0 atol = 1e-12            # (∫ρ - 0.4|Ω|)/(0.4|Ω|) ≤ 0  ⇔  aᵀz ≤ 1
    # a projected volume is not linear in z
    Vp = Functional(θ -> sum(∫(θ.ρ)q.dΩ) - 0.4, θ -> (δ -> ∫(δ)q.dΩ), q.θ; scale = 0.4)
    bad = Optimization(q.z; objective = q.C, linear_constraints = (Vp,))
    @test_throws ArgumentError solve!(bad, SIMP(OC(); metric = Euclidean()); maxiter = 2, on_failure = RejectStep(shrink = 0.5, max = 3))
    @test FunctionalOracle(q.problem, Euclidean()).metric === nothing
  end

  @testset "Problem and method arguments" begin
    q = compliance_problem()
    @test_throws ArgumentError Optimization(q.z; objective = q.C, constraints = (q.Vol,))
    @test_throws ArgumentError Optimization(q.z; objective = q.C, linear_constraints = (q.Vol, q.Vol))
    @test_throws ArgumentError Optimization(q.z; objective = q.C, linear_constraints = (q.C,))   # has states
    @test_throws UndefKeywordError SIMP(OC())
    @test_throws ArgumentError SIMP(OC(); metric = :lumped)
    @test_throws UndefKeywordError solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 2)
    @test_throws ArgumentError RejectStep(shrink = 1.5, max = 2)
    @test_throws ArgumentError StopCriteria(change = -1)
    @test_throws ArgumentError Continuation(β = q.H => [1, 2], γ = q.H => [1]; every = 2)
    @test_throws ArgumentError Continuation(β = 3 => [1, 2]; every = 2)
    @test HyperShape.solve! === Gridap.solve!
    ctx = q.problem.ctx
    @test_throws ArgumentError (ctx.iteration = 3)
    @test_throws ArgumentError OptContext(:other)
  end

  @testset "OC and SiMPL decrease the objective and keep the volume ($(design))" for design in (:P0, :P1)
    for opt in (OC(), SiMPL())
      q = compliance_problem(; design)
      metric = opt isa OC ? Euclidean() : LumpedMass(q.dΩ)
      seen = Tuple[]
      r = solve!(q.problem, SIMP(opt; metric); maxiter = 15, on_failure = RejectStep(shrink = 0.5, max = 3),
                 post = ctx -> push!(seen, (ctx.iteration, ctx.attempt, ctx.phase)))
      h = r.history
      @test r.status === :maxiter && r.iterations == 15 && length(h) == 15
      @test h.objective[end] < 0.9 * h.objective[1]
      @test all(<=(2e-3), h.constraint)                      # OC: up to its bisection tolerance
      @test r.objective ≈ h.objective[end]
      @test first.(seen) == 1:15 && all(s -> s[3] === :optimization, seen)
      opt isa SiMPL && @test nonincreasing(h.objective)
      # solves per iteration (after the first): self-adjoint, so no adjoint
      @test all(==(0), h.adjoint)
      if opt isa OC
        @test all(==(1), h.forward[2:end]) && all(==(2), h.filter[2:end])
      else
        @test all(h.filter[2:end] .== h.forward[2:end] .+ 1)
      end
      @test any(c -> c[2] === q.problem.ctx, q.ctxs)        # the user's solve receives the context of the problem
    end
  end

  @testset "Functional with an adjoint" begin
    q = compliance_problem()
    (; z, θ, S, A, dk, dΩ, dΓ) = q
    B = Functional((u, θ) -> ∫(u)dΓ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), S, A, θ; scale = :initial)
    r = solve!(Optimization(z; objective = B, linear_constraints = q.Vol), SIMP(OC(); metric = Euclidean());
               maxiter = 6, on_failure = RejectStep(shrink = 0.5, max = 3))
    @test all(==(1), r.history.adjoint[2:end])
    @test r.history.objective[end] < r.history.objective[1]
  end

  @testset "Rejected steps" begin
    # the third forward solve fails once
    cnt = Ref(0)
    q = compliance_problem(; solve_flag = () -> (cnt[] += 1; cnt[] != 3))
    r = solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 6, on_failure = RejectStep(shrink = 0.5, max = 3))
    h = r.history
    k = findfirst(>(1), h.attempts)
    @test k !== nothing && h.attempts[k] == 2
    @test h.forward[k] == 2                     # the failed design and the retried one; the old design comes from the snapshot
    @test h.step[k] ≈ 0.1 && h.step[k+1] ≈ 0.2   # move limit shrunk once, restored after the step stands
    @test r.status === :maxiter
    # every design fails after the second solve: abort, z at the last accepted design
    cnt[] = 0
    q = compliance_problem(; solve_flag = () -> (cnt[] += 1; cnt[] <= 2))
    accepted = Vector{Float64}[]
    @test_throws OptimizationFailure solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 6,
                                            on_failure = RejectStep(shrink = 0.5, max = 2),
                                            post = ctx -> push!(accepted, copy(get_values(q.z))))
    @test get_values(q.z) == last(accepted)
    # the forward problem fails at the start
    q = compliance_problem(; solve_flag = () -> false)
    err = try
      solve!(q.problem, SIMP(SiMPL(); metric = Euclidean()); maxiter = 3, on_failure = RejectStep(shrink = 0.5, max = 2))
      nothing
    catch e
      e
    end
    @test err isa OptimizationFailure && occursin("first iteration", sprint(showerror, err))
  end

  @testset "OC: the move limit is restored after a standing step" begin
    o = FunctionOracle(z -> sum(abs2, z .- 0.9) / 2, (d, z) -> (d .= z .- 0.9); constraint = (ones(4), 1.6))
    opt = OC()
    zz = fill(0.4, 4)
    s = init(opt, zz, o)
    shrink!(opt, s, 0.5)
    @test step!(opt, s, zz, o, nothing).step ≈ 0.1
    @test step!(opt, s, zz, o, nothing).step ≈ 0.2
  end

  @testset "Continuation and stopping" begin
    q = compliance_problem()
    set_beta!(q.H, 3.0)
    sched = Continuation(β = q.H => [1.0, 2.0, 4.0]; every = 3)
    r = solve!(q.problem, SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 12, schedule = sched,
               on_failure = RejectStep(shrink = 0.5, max = 3))
    @test vec(r.history.continuation)[1:9] == [1, 1, 1, 2, 2, 2, 4, 4, 4]
    @test q.H.β == 4.0 && r.history.names == [:β]
    # a criterion always met: one iteration per stage, stop in the last one
    q = compliance_problem()
    r = solve!(q.problem, SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 12, stop = StopCriteria(change = 1.0),
               schedule = Continuation(β = q.H => [1.0, 2.0, 4.0]; every = 3), on_failure = RejectStep(shrink = 0.5, max = 3))
    @test r.status === :converged && r.reason === :change && r.iterations == 3
    @test vec(r.history.continuation) == [1, 2, 4]
    # kkt with an optimiser that does not compute it
    q = compliance_problem()
    @test_throws ArgumentError solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 3, stop = StopCriteria(kkt = 1e-3),
                                      on_failure = RejectStep(shrink = 0.5, max = 3))
    q = compliance_problem()
    r = solve!(q.problem, SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 200, stop = StopCriteria(objective = 1e-4, window = 3),
               on_failure = RejectStep(shrink = 0.5, max = 3))
    @test r.status === :converged && r.reason === :objective && r.iterations < 200
  end

  @testset "Checkpoint and resume" begin
    dir = mktempdir()
    run(q, maxiter; kwargs...) = solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter, on_failure = RejectStep(shrink = 0.5, max = 3), kwargs...)
    qa = compliance_problem()
    ra = run(qa, 8)
    qb = compliance_problem()
    rb = run(qb, 4; checkpoint = Checkpoint(dir; every = 2))
    @test isfile(joinpath(dir, "checkpoint.toml")) && isfile(joinpath(dir, "z.bin")) && isfile(joinpath(dir, "history.csv"))
    ck = load_checkpoint(dir)
    @test ck.iteration == 4 && ck.z == get_values(qb.z) && length(ck.history.objective) == 4
    qc = compliance_problem()
    rc = run(qc, 8; resume = ck)
    @test get_values(qc.z) ≈ get_values(qa.z) rtol = 1e-12   # OC resumes exactly (up to threaded sums)
    @test isapprox(rc.history.objective, ra.history.objective; rtol = 1e-12) && rc.history.iteration == 1:8
    @test qc.C.scale[] == qa.C.scale[]                    # the scale fixed in the first run
    # SiMPL resumes (not bit-identical) and the checks on the checkpoint
    qd = compliance_problem()
    rd = solve!(qd.problem, SIMP(SiMPL(); metric = LumpedMass(qd.dΩ)); maxiter = 6, resume = ck,
                on_failure = RejectStep(shrink = 0.5, max = 3))
    @test rd.iterations == 6 && rd.history.objective[end] <= rd.history.objective[5] * (1 + 1e-12)
    qe = compliance_problem(; design = :P1)
    @test_throws ArgumentError run(qe, 8; resume = ck)
    @test_throws ArgumentError run(compliance_problem(), 4; resume = ck)   # already at maxiter
  end

  @testset "3D ($(design))" for design in (:P0, :P1)
    q = compliance_problem(; design, dim = 3, n = 4)
    @test check_consistency(q.C, q.Vol).passed
    r = solve!(q.problem, SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 5, on_failure = RejectStep(shrink = 0.5, max = 3))
    @test r.history.objective[end] < r.history.objective[1] && nonincreasing(r.history.objective)
  end
end
