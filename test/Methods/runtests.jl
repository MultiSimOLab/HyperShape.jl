using Gridap
using Gridap.FESpaces
using HyperFEM
using HyperShape
using HyperShape.Variables: get_values, set_values!, set_beta!
using HyperShape.Operators: num_solves
using HyperShape.Optimizers: FunctionOracle, OC, SiMPL, init, step!, shrink!
using HyperShape.Methods: FunctionalOracle, _setup_constraint!, OptHistory, _record!, _stop_reason
using LinearAlgebra
using Test

# heat compliance (self-adjoint) and the volume of the unprojected design density
function compliance_problem(; design = :P0, n = 8, dim = 2, init = 0.4, fraction = 0.4, solve_flag = nothing,
                            phases = nothing, sign = Ref(1.0))
  p = state_problem(; design, n, dim, init)
  (; z, x, θ, S, f, dk, dΩ) = p
  flag = solve_flag === nothing ? () -> true : solve_flag
  Sc = (solve_flag === nothing && phases === nothing) ? S :
       StateOperator(S.model; depends = θ, solve = (m, ctx) -> begin
         push!(p.ctxs, (:S, ctx))
         phases === nothing || push!(phases, (ctx.iteration, ctx.trial, ctx.phase))
         (solve!(m; Assembly = true), flag())
       end)
  C = Functional((u, θ) -> ∫(f * u)dΩ, (u, w, θ) -> (δ -> ∫(-sign[] * (dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), Sc, Sc, θ; scale = :initial)
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
    # a uniform start at 0.3 used to hide a nonlinear constraint
    q3 = compliance_problem(; design, init = 0.3)
    Vp3 = Functional(θ -> sum(∫(θ.ρ)q3.dΩ) - 0.4, θ -> (δ -> ∫(δ)q3.dΩ), q3.θ; scale = 0.4)
    @test_throws ArgumentError solve!(Optimization(q3.z; objective = q3.C, linear_constraints = Vp3),
                                      SIMP(OC(); metric = Euclidean()); maxiter = 2, on_failure = RejectStep(shrink = 0.5, max = 3))
    @test get_values(q3.z) == fill(0.3, length(q3.z))     # z restored after the probes
    # the filtered, unprojected volume is linear and has the same vector (Neumann filter)
    θf = PhysicalFields(q.x; ρ = HelmholtzFilter(q.Sf; measure = q.dΩ))
    Vf = Functional(θ -> sum(∫(θ.ρ)q.dΩ) - 0.4 * sum(m), θ -> (δ -> ∫(δ)q.dΩ), θf; scale = 0.4 * sum(m))
    of = FunctionalOracle(Optimization(q.z; objective = q.C, linear_constraints = Vf), Euclidean())
    _setup_constraint!(of, Vf, q.problem.ctx)
    @test of.a ≈ m ./ (0.4 * sum(m)) rtol = 1e-10
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
    @test isfile(joinpath(dir, "checkpoint.toml")) && isfile(joinpath(dir, "z_4.bin")) && isfile(joinpath(dir, "history_4.csv"))
    @test !isfile(joinpath(dir, "z_2.bin"))                # older files deleted
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

  @testset "Stalled: z, fields and state at the last accepted design" begin
    sign = Ref(1.0)
    q = compliance_problem(; sign)
    r = solve!(q.problem, SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 10,
               on_failure = RejectStep(shrink = 0.5, max = 3), post = ctx -> (ctx.iteration == 2 && (sign[] = -1.0)))
    @test r.status === :stalled && r.iterations == 2
    @test get_values(q.z) == r.state.z || get_values(q.z) ≈ r.state.z
    @test get_free_dof_values(q.x.ρ) == get_values(q.z)
    @test functional_value(q.C, q.problem.ctx) ≈ r.objective
  end

  @testset "Failures: state, fields and checkpoint" begin
    dir = mktempdir()
    cnt = Ref(0)
    q = compliance_problem(; solve_flag = () -> (cnt[] += 1; cnt[] <= 2))
    accepted = Vector{Float64}[]
    @test_throws OptimizationFailure solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 6,
                                            on_failure = RejectStep(shrink = 0.5, max = 2), checkpoint = Checkpoint(dir; every = 10),
                                            post = ctx -> push!(accepted, copy(get_values(q.z))))
    @test get_free_dof_values(q.x.ρ) == get_values(q.z) == last(accepted)
    ck = load_checkpoint(dir)
    @test ck.iteration == 1 && ck.z == last(accepted)
    # an adjoint that fails: non-finite gradient, checkpoint written
    q = compliance_problem()
    (; z, θ, S, dk, dΩ, dΓ, A) = q
    Abad = StateOperator(A.model; depends = (θ, S), solve = (m, ctx) -> (solve!(m; Assembly = true), false))
    B = Functional((u, θ) -> ∫(u)dΓ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), S, Abad, θ; scale = 1.0)
    dir2 = mktempdir()
    @test_throws OptimizationFailure solve!(Optimization(z; objective = B, linear_constraints = q.Vol),
                                            SIMP(OC(); metric = Euclidean()); maxiter = 3,
                                            on_failure = RejectStep(shrink = 0.5, max = 2), checkpoint = Checkpoint(dir2; every = 10))
    @test isfile(joinpath(dir2, "checkpoint.toml"))
  end

  @testset "Phases seen by the user's solve (SiMPL, non-uniform start, jump)" begin
    phases = Tuple{Int,Int,Symbol}[]
    q = compliance_problem(; init = nothing, fraction = 0.7, phases)
    r = solve!(q.problem, SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 5,
               schedule = Continuation(β = q.H => [2.0, 4.0]; every = 3), on_failure = RejectStep(shrink = 0.5, max = 3))
    @test r.iterations == 5
    # the first solve of iteration 1 and of the jump iteration are the current design
    first_of(k) = phases[findfirst(p -> p[1] == k, phases)]
    @test first_of(1)[3] === :optimization
    @test first_of(4)[3] === :optimization
    @test all(p -> p[3] in (:setup, :optimization, :linesearch), phases)
  end

  @testset "Resume in the middle of a continuation stage" begin
    dir = mktempdir()
    sched(q) = Continuation(β = q.H => [1.0, 2.0, 4.0]; every = 3)
    run(q, maxiter; kwargs...) = solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter, schedule = sched(q),
                                        on_failure = RejectStep(shrink = 0.5, max = 3), kwargs...)
    qa = compliance_problem(); ra = run(qa, 9)
    qb = compliance_problem(); run(qb, 5; checkpoint = Checkpoint(dir; every = 5))
    ck = load_checkpoint(dir)
    @test ck.stage == 2 && ck.stage_iterations == 2
    qc = compliance_problem(); rc = run(qc, 9; resume = ck)
    @test get_values(qc.z) ≈ get_values(qa.z) rtol = 1e-12
    @test vec(rc.history.continuation) == vec(ra.history.continuation)
    qd = compliance_problem()
    @test_throws ArgumentError solve!(qd.problem, SIMP(OC(); metric = Euclidean()); maxiter = 9, resume = ck,
                                      schedule = Continuation(β = qd.H => [1.0]; every = 3), on_failure = RejectStep(shrink = 0.5, max = 3))
  end

  @testset "Objective criterion counts only the current stage" begin
    q = compliance_problem()
    r = solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 40, stop = StopCriteria(objective = 5e-2, window = 2),
               schedule = Continuation(β = q.H => [1.0, 2.0, 4.0, 8.0]; every = 20), on_failure = RejectStep(shrink = 0.5, max = 3))
    c = vec(r.history.continuation)
    lengths = [count(==(v), c) for v in unique(c)]
    @test all(>=(3), lengths[1:end-1])          # window + 1 iterations at least before a stage can end
    @test_throws ArgumentError StopCriteria(objective = 1e-3)
  end

  @testset "SiMPL without constraint, loop bookkeeping without allocations" begin
    q = compliance_problem()
    r = solve!(Optimization(q.z; objective = q.C), SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 4,
               on_failure = RejectStep(shrink = 0.5, max = 3))
    @test r.history.objective[end] < r.history.objective[1] && all(isnan, r.history.constraint)
    h = OptHistory(10, (:β,))
    buf = [1.0]
    row = (1, 1.0, 0.0, NaN, 0.1, 0.2, 1, 1, 0, 2)
    _record!(h, row, buf)
    @test (@allocated _record!(h, row, buf)) == 0
    st = StopCriteria(change = 1e-3, objective = 1e-3, window = 1)
    info = HyperShape.Optimizers.StepInfo(true, 1.0, 1.0, NaN, 0.1, 0.1, 0.0, 1)
    _stop_reason(st, info, h, 0)
    @test (@allocated _stop_reason(st, info, h, 0)) == 0
    @test_throws ArgumentError Continuation(step = q.H => [1.0]; every = 2)    # history column name
    @test_throws ArgumentError Continuation(β = q.H => [1.0, -2.0]; every = 2)
  end
end
