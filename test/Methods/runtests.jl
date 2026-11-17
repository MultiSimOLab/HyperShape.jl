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

# an optimiser that does not compute the stationarity measure: SiMPL with the 9-argument StepInfo
struct NoStatOpt <: HyperShape.Optimizers.Optimizer end
const _NS = HyperShape.Optimizers
_NS.init(::NoStatOpt, z, oracle) = _NS.init(SiMPL(), z, oracle)
function _NS.step!(::NoStatOpt, state, z, oracle, ctx)
  i = _NS.step!(SiMPL(), state, z, oracle, ctx)
  _NS.StepInfo(i.accepted, i.f, i.f_new, i.kkt, i.change, i.step, i.violation, i.evaluations, i.multiplier)
end
_NS.shrink!(::NoStatOpt, state, f) = _NS.shrink!(SiMPL(), state, f)
_NS.reset!(::NoStatOpt, state) = _NS.reset!(SiMPL(), state)
_NS.constraint_tol(::NoStatOpt) = _NS.constraint_tol(SiMPL())

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
  C = Functional((u, θ) -> ∫(f * u)dΩ, Sc; derivative = (u, w, θ) -> (δ -> ∫(-sign[] * (dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = :self, fields = θ, scale = :initial)
  θv = PhysicalFields(x; ρ = Identity())
  vol = sum(∫(1.0)dΩ)
  Vol = Functional(θ -> sum(∫(θ.ρ)dΩ) - fraction * vol, θv; derivative = θ -> (δ -> ∫(δ)dΩ), scale = fraction * vol, linear = true)
  (; p..., C, Vol, θv, S = Sc, problem = Optimization(z; objective = C, constraints = (Vol,)))
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
    Vp = Functional(θ -> sum(∫(θ.ρ)q.dΩ) - 0.4, q.θ; derivative = θ -> (δ -> ∫(δ)q.dΩ), scale = 0.4, linear = true)
    bad = Optimization(q.z; objective = q.C, constraints = (Vp,))
    @test_throws ArgumentError solve!(bad, SIMP(OC(); metric = Euclidean()); maxiter = 2, verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
    @test FunctionalOracle(q.problem, Euclidean()).metric === nothing
    # a uniform start at 0.3 used to hide a nonlinear constraint
    q3 = compliance_problem(; design, init = 0.3)
    Vp3 = Functional(θ -> sum(∫(θ.ρ)q3.dΩ) - 0.4, q3.θ; derivative = θ -> (δ -> ∫(δ)q3.dΩ), scale = 0.4, linear = true)
    @test_throws ArgumentError solve!(Optimization(q3.z; objective = q3.C, constraints = Vp3),
                                      SIMP(OC(); metric = Euclidean()); maxiter = 2, verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
    @test get_values(q3.z) == fill(0.3, length(q3.z))     # z restored after the probes
    # the filtered, unprojected volume is linear and has the same vector (Neumann filter)
    θf = PhysicalFields(q.x; ρ = HelmholtzFilter(q.Sf, q.dΩ))
    Vf = Functional(θ -> sum(∫(θ.ρ)q.dΩ) - 0.4 * sum(m), θf; derivative = θ -> (δ -> ∫(δ)q.dΩ), scale = 0.4 * sum(m), linear = true)
    of = FunctionalOracle(Optimization(q.z; objective = q.C, constraints = Vf), Euclidean())
    _setup_constraint!(of, Vf, q.problem.ctx)
    @test of.a ≈ m ./ (0.4 * sum(m)) rtol = 1e-10
  end

  @testset "Problem and method arguments" begin
    q = compliance_problem()
    @test Optimization(q.z; objective = q.C, constraints = q.Vol).constraints === (q.Vol,)
    @test_throws ArgumentError Optimization(q.z; objective = q.C, constraints = (q.Vol, q.Vol))
    @test_throws ArgumentError Optimization(q.z; objective = q.C, constraints = (q.C,))   # has states
    @test SIMP(OC()).metric isa Euclidean           # default metric of OC (which does not use it)
    @test_throws ArgumentError SIMP(SiMPL())         # SiMPL needs one
    @test_throws ArgumentError SIMP(OC(); metric = :lumped)
    @test_throws UndefKeywordError solve!(q.problem, SIMP(OC()))   # maxiter is mandatory
    @test_throws ArgumentError solve!(q.problem, SIMP(OC()); maxiter = 2, on_failure = :reject)
    @test_throws ArgumentError solve!(q.problem, SIMP(OC()); maxiter = 2, on_failure = nothing)
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
      r = solve!(q.problem, SIMP(opt; metric); maxiter = 15, verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3),
                 post = OptPostProcessor((_, ctx) -> push!(seen, (ctx.iteration, ctx.attempt, ctx.phase)); every = 1, initial = false), stop = nothing)
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
    B = Functional((u, θ) -> ∫(u)dΓ, S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = A, fields = θ, scale = :initial)
    r = solve!(Optimization(z; objective = B, constraints = q.Vol), SIMP(OC(); metric = Euclidean());
               maxiter = 6, verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
    @test all(==(1), r.history.adjoint[2:end])
    @test r.history.objective[end] < r.history.objective[1]
  end

  @testset "Rejected steps" begin
    # the third forward solve fails once
    cnt = Ref(0)
    q = compliance_problem(; solve_flag = () -> (cnt[] += 1; cnt[] != 3))
    # scalar move limit (adapt = nothing): exact values of the step; the adaptive case is tested in Optimizers
    r = solve!(q.problem, SIMP(OC(adapt = nothing); metric = Euclidean()); maxiter = 6, verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
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
                                            verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 2),
                                            post = OptPostProcessor((_, ctx) -> push!(accepted, copy(get_values(q.z))); every = 1, initial = false), stop = nothing)
    @test get_values(q.z) == last(accepted)
    # the forward problem fails at the start
    q = compliance_problem(; solve_flag = () -> false)
    err = try
      solve!(q.problem, SIMP(SiMPL(); metric = Euclidean()); maxiter = 3, verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 2), stop = nothing)
      nothing
    catch e
      e
    end
    @test err isa OptimizationFailure && occursin("first iteration", sprint(showerror, err))
    @test err.iterations == 0 && isempty(err.history.iteration) && size(err.history.continuation, 2) == 0
  end

  @testset "OC: move limit per continuation stage, banner" begin
    for adapt in (nothing, (0.7, 1.2))
      q = compliance_problem()
      oc = OC(; adapt)
      io = IOBuffer()
      r = solve!(q.problem, SIMP(oc); maxiter = 6, stop = nothing, io, verbose = 1,
                 schedule = Continuation(β = q.H => [1.0, 2.0, 4.0], move = oc => [0.2, 0.1, 0.05]; every = 2))
      st = r.history.step
      if adapt === nothing
        @test st == [0.2, 0.2, 0.1, 0.1, 0.05, 0.05]
      else
        @test all(st[1:2] .<= 0.2 + 1e-12) && all(st[3:4] .<= 0.1 + 1e-12) && all(st[5:6] .<= 0.05 + 1e-12)
        @test st[3] ≈ 0.1 && st[5] ≈ 0.05      # restored to the new ceiling at the jump
      end
      @test oc.move == 0.2                                    # the move of the OC is restored at the end
      out = String(take!(io))
      @test occursin(adapt === nothing ? "OC: move 0.2 · fixed (top88)" : "adaptive per entry ×0.7/×1.2, floor 0.001", out)
    end
    @test_throws ArgumentError Continuation(move = OC() => [0.2, -0.1]; every = 2)
    @test_throws ArgumentError Continuation(move = OC() => [0.2, 5e-4]; every = 2)   # below move_min
    q = compliance_problem()
    @test_throws ArgumentError solve!(q.problem, SIMP(OC()); maxiter = 2, verbose = 0,
                                      schedule = Continuation(move = OC() => [0.2, 0.1]; every = 1))
    # the banner shows the move limit of the first stage
    oc = OC()
    io = IOBuffer()
    solve!(q.problem, SIMP(oc); maxiter = 1, stop = nothing, io, verbose = 1,
           schedule = Continuation(move = oc => [0.1, 0.05]; every = 2))
    @test occursin("OC: move 0.1 ·", String(take!(io)))
    @test_throws ArgumentError (OC().move = 0.1)
    @test_throws ArgumentError OC(adapt = (1.2, 0.7))
    @test_throws ArgumentError OC(move_min = 0.5)
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
               verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
    @test vec(r.history.continuation)[1:9] == [1, 1, 1, 2, 2, 2, 4, 4, 4]
    @test q.H.β == 4.0 && r.history.names == [:β]
    # a criterion always met: one iteration per stage, stop in the last one
    q = compliance_problem()
    r = solve!(q.problem, SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 12, stop = StopCriteria(change = 1.0),
               schedule = Continuation(β = q.H => [1.0, 2.0, 4.0]; every = 3, close_early = true), verbose = 0,
               on_failure = RejectStep(shrink = 0.5, max = 3))
    @test r.status === :converged && r.reason === :change && r.iterations == 6   # not at the first iteration of a stage
    @test vec(r.history.continuation) == [1, 1, 2, 2, 4, 4]
    # by default the criteria only act in the last stage: the earlier stages run their `every` iterations
    q = compliance_problem()
    r = solve!(q.problem, SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 12, stop = StopCriteria(change = 1.0),
               schedule = Continuation(β = q.H => [1.0, 2.0, 4.0]; every = 3), verbose = 0)
    @test r.status === :converged && r.iterations == 8 && vec(r.history.continuation) == [1, 1, 1, 2, 2, 2, 4, 4]
    # kkt with an optimiser that does not compute it
    q = compliance_problem()
    @test_throws ArgumentError solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 3, stop = StopCriteria(kkt = 1e-3),
                                      verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3))
    q = compliance_problem()
    r = solve!(q.problem, SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 200, stop = StopCriteria(objective = 1e-4, window = 3),
               verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3))
    @test r.status === :converged && r.reason === :objective && r.iterations < 200
    # stationarity measure: recorded by SiMPL and OC, absolute and relative criteria
    for opt in (SiMPL(), OC())
      q = compliance_problem()
      r = solve!(q.problem, SIMP(opt; metric = LumpedMass(q.dΩ)); maxiter = 8, stop = nothing, verbose = 0,
                 on_failure = RejectStep(shrink = 0.5, max = 3))
      S = r.history.stationarity
      @test length(S) == r.iterations && all(isfinite, S) && all(>(0), S)
      q = compliance_problem()
      r2 = solve!(q.problem, SIMP(opt; metric = LumpedMass(q.dΩ)); maxiter = 50, verbose = 0,
                  stop = StopCriteria(stationarity = 2 * minimum(S)), on_failure = RejectStep(shrink = 0.5, max = 3))
      @test r2.status === :converged && r2.reason === :stationarity && r2.iterations <= 8
      q = compliance_problem()
      r3 = solve!(q.problem, SIMP(opt; metric = LumpedMass(q.dΩ)); maxiter = 50, verbose = 0,
                  stop = StopCriteria(stationarity_rel = 2 * minimum(S) / S[1]), on_failure = RejectStep(shrink = 0.5, max = 3))
      @test r3.status === :converged && r3.reason === :stationarity_rel && r3.iterations <= 8
    end
    @test_throws ArgumentError StopCriteria(stationarity = -1.0)
    @test_throws ArgumentError StopCriteria(stationarity_rel = -1.0)
    # OC and SiMPL give the same S at the same design and metric
    q1, q2 = compliance_problem(), compliance_problem()
    r1 = solve!(q1.problem, SIMP(OC(); metric = LumpedMass(q1.dΩ)); maxiter = 1, stop = nothing, verbose = 0)
    r2 = solve!(q2.problem, SIMP(SiMPL(); metric = LumpedMass(q2.dΩ)); maxiter = 1, stop = nothing, verbose = 0)
    @test r1.history.stationarity[1] ≈ r2.history.stationarity[1] rtol = 1e-10
    # an optimiser that does not compute it (9-argument StepInfo): error when a criterion asks for it
    q = compliance_problem()
    @test_throws ArgumentError solve!(q.problem, SIMP(NoStatOpt(); metric = LumpedMass(q.dΩ)); maxiter = 3, verbose = 0,
                                      stop = StopCriteria(stationarity = 1e-3))
    q = compliance_problem()
    rn = solve!(q.problem, SIMP(NoStatOpt(); metric = LumpedMass(q.dΩ)); maxiter = 2, verbose = 0, stop = nothing)
    @test all(isnan, rn.history.stationarity)
  end

  @testset "Checkpoint and resume" begin
    dir = mktempdir()
    run(q, maxiter; kwargs...) = solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter, verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), kwargs..., stop = nothing)
    qa = compliance_problem()
    ra = run(qa, 8)
    qb = compliance_problem()
    rb = run(qb, 4; checkpoint = Checkpoint(dir; every = 2))
    @test isfile(joinpath(dir, "checkpoint.toml")) && isfile(joinpath(dir, "z_4.bin")) && isfile(joinpath(dir, "history_4.csv"))
    @test !isfile(joinpath(dir, "z_2.bin")) && !isfile(joinpath(dir, "memory_2.bin"))   # older files deleted
    ck = load_checkpoint(dir)
    @test ck.optimizer == "OC" && length(ck.memory) == 3length(ck.z) + 2   # adaptive move limits saved
    @test ck.iteration == 4 && ck.z == get_values(qb.z) && length(ck.history.objective) == 4
    qc = compliance_problem()
    rc = run(qc, 8; resume = ck)
    @test get_values(qc.z) ≈ get_values(qa.z) rtol = 1e-12   # OC with adaptive limits resumes exactly (up to threaded sums)
    @test isapprox(rc.history.objective, ra.history.objective; rtol = 1e-12) && rc.history.iteration == 1:8
    @test qc.C.scale[] == qa.C.scale[]                    # the scale fixed in the first run
    # SiMPL resumes (not bit-identical) and the checks on the checkpoint
    qd = compliance_problem()
    rd = solve!(qd.problem, SIMP(SiMPL(); metric = LumpedMass(qd.dΩ)); maxiter = 6, resume = ck,
                verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
    @test rd.iterations == 6 && rd.history.objective[end] <= rd.history.objective[5] * (1 + 1e-12)
    qe = compliance_problem(; design = :P1)
    @test_throws ArgumentError run(qe, 8; resume = ck)
    @test_throws ArgumentError run(compliance_problem(), 4; resume = ck)   # already at maxiter
  end

  @testset "3D ($(design))" for design in (:P0, :P1)
    q = compliance_problem(; design, dim = 3, n = 4)
    @test check_consistency(q.C, q.Vol).passed
    r = solve!(q.problem, SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 5, verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
    @test r.history.objective[end] < r.history.objective[1] && nonincreasing(r.history.objective)
  end

  @testset "Stalled: z, fields and state at the last accepted design" begin
    sign = Ref(1.0)
    q = compliance_problem(; sign)
    r = solve!(q.problem, SIMP(SiMPL(max_backtracks = 1); metric = LumpedMass(q.dΩ)); maxiter = 10,
               verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), post = OptPostProcessor((_, ctx) -> (ctx.iteration == 2 && (sign[] = -1.0)); every = 1, initial = false), stop = nothing)
    @test r.status === :stalled && r.iterations == 2
    @test get_values(q.z) == r.state.z
    @test get_free_dof_values(q.x.ρ) == get_values(q.z)
    @test functional_value(q.C, q.problem.ctx) ≈ r.objective
  end

  @testset "Failures: state, fields and checkpoint" begin
    dir = mktempdir()
    cnt = Ref(0)
    q = compliance_problem(; solve_flag = () -> (cnt[] += 1; cnt[] <= 2))
    accepted = Vector{Float64}[]
    err = try
      solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 6, verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 2),
             checkpoint = Checkpoint(dir; every = 10), post = OptPostProcessor((_, ctx) -> push!(accepted, copy(get_values(q.z))); every = 1, initial = false), stop = nothing)
      nothing
    catch e
      e
    end
    @test err isa OptimizationFailure
    @test err.iterations == 1 && length(err.history) == 1 && err.history.iteration == [1]
    @test get_free_dof_values(q.x.ρ) == get_values(q.z) == last(accepted)
    # every solve fails now, also the one at the last accepted design, and the message says so
    @test !HyperShape.Operators.is_converged(q.S) && occursin("could not be solved again", sprint(showerror, err))
    ck = load_checkpoint(dir)
    @test ck.iteration == 1 && ck.z == last(accepted)
    # an adjoint that fails: non-finite gradient, checkpoint written
    q = compliance_problem()
    (; z, θ, S, dk, dΩ, dΓ, A) = q
    Abad = StateOperator(A.model; depends = (θ, S), solve = (m, ctx) -> (solve!(m; Assembly = true), false))
    B = Functional((u, θ) -> ∫(u)dΓ, S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = Abad, fields = θ, scale = 1.0)
    dir2 = mktempdir()
    @test_throws OptimizationFailure solve!(Optimization(z; objective = B, constraints = q.Vol),
                                            SIMP(OC(); metric = Euclidean()); maxiter = 3,
                                            verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 2), checkpoint = Checkpoint(dir2; every = 10), stop = nothing)
    @test isfile(joinpath(dir2, "checkpoint.toml"))
  end

  @testset "Phases seen by the user's solve (SiMPL, non-uniform start, jump)" begin
    phases = Tuple{Int,Int,Symbol}[]
    q = compliance_problem(; init = nothing, fraction = 0.7, phases)
    r = solve!(q.problem, SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 5,
               schedule = Continuation(β = q.H => [2.0, 4.0]; every = 3), verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
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
                                        verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), kwargs..., stop = nothing)
    qa = compliance_problem(); ra = run(qa, 9)
    qb = compliance_problem(); run(qb, 5; checkpoint = Checkpoint(dir; every = 5))
    ck = load_checkpoint(dir)
    @test ck.stage == 2 && ck.stage_iterations == 2
    qc = compliance_problem(); rc = run(qc, 9; resume = ck)
    @test get_values(qc.z) ≈ get_values(qa.z) rtol = 1e-12
    @test vec(rc.history.continuation) == vec(ra.history.continuation)
    qd = compliance_problem()
    @test_throws ArgumentError solve!(qd.problem, SIMP(OC(); metric = Euclidean()); maxiter = 9, resume = ck,
                                      schedule = Continuation(β = qd.H => [1.0]; every = 3), verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
  end

  @testset "Objective criterion counts only the current stage" begin
    q = compliance_problem()
    r = solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 40, stop = StopCriteria(objective = 5e-2, window = 2),
               schedule = Continuation(β = q.H => [1.0, 2.0, 4.0, 8.0]; every = 20, close_early = true), verbose = 0,
               on_failure = RejectStep(shrink = 0.5, max = 3))
    c = vec(r.history.continuation)
    lengths = [count(==(v), c) for v in unique(c)]
    @test all(>=(2), lengths[1:end-1])          # window iterations at least before a stage can end
    @test StopCriteria(objective = 1e-3).window == 10
    @test_throws ArgumentError StopCriteria(objective = 1e-3, window = 1)
  end

  @testset "SiMPL without constraint, loop bookkeeping without allocations" begin
    q = compliance_problem()
    r = solve!(Optimization(q.z; objective = q.C), SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 4,
               verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
    @test r.history.objective[end] < r.history.objective[1] && all(isnan, r.history.constraint)
    @test all(isnan, r.history.multiplier) && r.start_scaling === nothing
    h = OptHistory(10, (:β,))
    buf = [1.0]
    row = (1, 1.0, 0.0, NaN, 0.1, 0.2, 1, 1, 0, 2, 0.5, NaN, NaN)
    _record!(h, row, buf)
    @test (@allocated _record!(h, row, buf)) == 0
    st = StopCriteria(change = 1e-3, objective = 1e-3, window = 2)
    info = HyperShape.Optimizers.StepInfo(true, 1.0, 1.0, NaN, 0.1, 0.1, 0.0, 1, 0.0)
    _stop_reason(st, info, h, 0)
    @test (@allocated _stop_reason(st, info, h, 0)) == 0
    @test_throws ArgumentError Continuation(step = q.H => [1.0]; every = 2)    # history column name
    @test_throws ArgumentError Continuation(β = q.H => [1.0, -2.0]; every = 2)
  end

  @testset "More review cases" begin
    # a projected volume from a uniform 0.5 start with η = 0.5 on a finer mesh is not linear
    q = compliance_problem(; n = 40, init = 0.5)
    set_beta!(q.H, 3.0)
    Vp = Functional(θ -> sum(∫(θ.ρ)q.dΩ) - 0.4, q.θ; derivative = θ -> (δ -> ∫(δ)q.dΩ), scale = 0.4, linear = true)
    o = FunctionalOracle(Optimization(q.z; objective = q.C, constraints = Vp), Euclidean())
    @test_throws ArgumentError _setup_constraint!(o, Vp, q.problem.ctx)
    # a start with values at the bounds: the first solve of SiMPL is the current design
    phases = Tuple{Int,Int,Symbol}[]
    q = compliance_problem(; init = [isodd(i) ? 0.0 : 0.8 for i in 1:64], phases)
    solve!(q.problem, SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 1, verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
    @test phases[findfirst(p -> p[1] == 1, phases)][3] === :optimization
    # a resumed run stops where the uninterrupted one does
    dir = mktempdir()
    stopc = StopCriteria(objective = 2e-2, window = 2)
    runs(q, maxiter; kwargs...) = solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter, stop = stopc,
                                         verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), kwargs...)
    ra = runs(compliance_problem(), 40)
    @test ra.status === :converged
    kstop = ra.iterations
    runs(compliance_problem(), kstop - 1; checkpoint = Checkpoint(dir; every = 100))
    rc = runs(compliance_problem(), 40; resume = load_checkpoint(dir))
    @test rc.iterations == kstop
    # a jump whose first design fails is undone: β and the checkpoint stay in the previous stage
    hold = Ref{Any}(nothing)
    qf2 = compliance_problem(; solve_flag = () -> hold[].β != 2.0)
    hold[] = qf2.H
    dir2 = mktempdir()
    @test_throws OptimizationFailure solve!(qf2.problem, SIMP(OC(); metric = Euclidean()); maxiter = 6,
                                            schedule = Continuation(β = qf2.H => [1.0, 2.0]; every = 2),
                                            checkpoint = Checkpoint(dir2; every = 100), verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 2), stop = nothing)
    ck = load_checkpoint(dir2)
    @test ck.stage == 1 && ck.iteration == 2 && qf2.H.β == 1.0
    # a fixed scale is kept on resume
    q = compliance_problem()
    rq = solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 2, checkpoint = Checkpoint(dir; every = 100),
                verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
    q2 = compliance_problem()
    q2.Vol.scale[] = 7.0
    solve!(q2.problem, SIMP(OC(); metric = Euclidean()); maxiter = 3, resume = load_checkpoint(dir),
           verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
    @test q2.Vol.scale[] == 7.0
    # a user's file in the checkpoint folder is not deleted
    write(joinpath(dir, "z_notes.txt"), "keep")
    solve!(compliance_problem().problem, SIMP(OC(); metric = Euclidean()); maxiter = 2, checkpoint = Checkpoint(dir; every = 1),
           verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
    @test isfile(joinpath(dir, "z_notes.txt"))
  end

  @testset "Decisions of Jesús: stalls end a stage, infeasible starts, multipliers" begin
    # no step at β = 1 (wrong gradient from iteration 3): the stage ends, β goes to 2, and the run stalls there
    sign = Ref(1.0)
    q = compliance_problem(; sign)
    r = solve!(q.problem, SIMP(SiMPL(max_backtracks = 1); metric = LumpedMass(q.dΩ)); maxiter = 10,
               schedule = Continuation(β = q.H => [1.0, 2.0]; every = 5), verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3),
               post = OptPostProcessor((_, ctx) -> (ctx.iteration == 2 && (sign[] = -1.0)); every = 1, initial = false), stop = nothing)
    @test r.status === :stalled && r.iterations == 2 && q.H.β == 2.0
    # an infeasible start is scaled towards the lower bound and SiMPL runs
    q = compliance_problem(; init = 0.6)
    io = IOBuffer()
    r = @test_logs solve!(q.problem, SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 5, verbose = 0, io,
                          on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)        # level 0: no logs, nothing on io
    @test isempty(take!(io)) && r.start_scaling ≈ 0.4 / 0.6
    @test r.iterations == 5 && all(<=(1e-8), r.history.constraint)
    @test nonincreasing(r.history.objective)
    # multipliers in the history
    @test all(>=(0), r.history.multiplier)
    q = compliance_problem()
    r = solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 3, verbose = 0, on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
    @test all(>(0), r.history.multiplier)
  end

  @testset "Second review: stalls, solve counts, old checkpoints, scaled start" begin
    # a stall ends a stage and the run goes on; the summary counts every forward solve of the call
    q = compliance_problem(; n = 16)
    c0 = q.nforward[]
    io = IOBuffer()
    r = solve!(q.problem, SIMP(SiMPL(max_backtracks = 0); metric = LumpedMass(q.dΩ)); maxiter = 30, io,
               schedule = Continuation(β = q.H => [1.0, 2.0, 4.0, 8.0]; every = 6), on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
    out = String(take!(io))
    @test occursin("the stage ends", out)
    ends = findfirst("the stage ends", out)
    @test count(l -> occursin(r"^\s+\d+\s+\d\.\d{6}e[+-]\d\d", l), split(out[last(ends):end], '\n')) >= 1   # rows after it
    m = match(r"solves: forward (\d+)", out)
    @test parse(Int, m[1]) == q.nforward[] - c0
    @test !occursin(r"─\n\s+it .*\n\s+─+\n\s+─", out)                   # no header without rows
    # a checkpoint of format 1 (no multiplier column) is still read
    dir = mktempdir()
    q = compliance_problem()
    rb = solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 2, verbose = 0, checkpoint = Checkpoint(dir; every = 2),
                on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
    ck = load_checkpoint(dir)
    @test ck.history.multiplier == rb.history.multiplier
    @test ck.history.stationarity == rb.history.stationarity && all(isfinite, ck.history.stationarity)
    hfile = joinpath(dir, "history_2.csv")
    lines = split.(readlines(hfile), ",")
    js = findfirst(==("stationarity"), lines[1])
    lines = [deleteat!(copy(l), js) for l in lines]                     # format 4: no stationarity column
    j = findfirst(==("multiplier"), lines[1])
    write(hfile, join((join(deleteat!(copy(l), j), ",") for l in lines), "\n") * "\n")
    tfile = joinpath(dir, "checkpoint.toml")
    write(tfile, replace(read(tfile, String), "format = 5" => "format = 1"))
    ck1 = load_checkpoint(dir)
    @test all(isnan, ck1.history.multiplier) && ck1.history.objective == ck.history.objective
    @test all(isnan, ck1.history.stationarity)
    r1 = solve!(compliance_problem().problem, SIMP(OC(); metric = Euclidean()); maxiter = 4, verbose = 0, resume = ck1,
                on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
    @test r1.iterations == 4 && isnan(r1.history.multiplier[1]) && r1.history.multiplier[4] > 0
    # the scaled start leaves the entries the constraint does not act on
    q = compliance_problem()
    nz = length(get_values(q.z))
    a = ones(nz); a[1:5] .= 0
    lb = zeros(nz)
    zk = fill(0.6, nz)
    t = HyperShape.Methods._make_feasible!((C = q.Vol, a = a, V = 0.4 * sum(a), lb = lb, z = q.z), zk)
    @test t ≈ 0.4 / 0.6 && all(==(0.6), zk[1:5]) && dot(a, zk) ≈ 0.4 * sum(a)
    # printing at level 0 allocates nothing
    h = OptHistory(4, (:β,))
    _record!(h, (1, 1.0, 0.0, NaN, 0.1, 0.2, 1, 1, 0, 2, 0.5, NaN, NaN), [1.0])
    p0 = HyperShape.Methods._Printer(IOBuffer(), 0, nothing)
    HyperShape.Methods._print_row(p0, h); HyperShape.Methods._print_event(p0, "x", :red)
    @test (@allocated HyperShape.Methods._print_row(p0, h)) == 0
    @test (@allocated HyperShape.Methods._print_event(p0, "x", :red)) == 0
  end

  @testset "Volume of the physical density: nonlinear constraint ($(name))" for name in (:OC, :SiMPL)
    q = compliance_problem()
    vol = sum(∫(1.0)q.dΩ)
    V = Volume(q.θ, q.dΩ; max_fraction = 0.4)
    @test !V.linear && Volume(q.x, q.dΩ; max_fraction = 0.4).linear
    @test HyperShape.Variables.is_linear_field(q.θv, :ρ) && !HyperShape.Variables.is_linear_field(q.θ, :ρ)
    @test check_consistency(V).passed                           # Taylor test of the projected volume
    method = name === :OC ? SIMP(OC()) : SIMP(SiMPL(); metric = LumpedMass(q.dΩ))
    io = IOBuffer()
    r = solve!(Optimization(q.z; objective = q.C, constraints = V), method; maxiter = 30, io, verbose = 1,
               schedule = Continuation(β = q.H => [3.0, 8.0]; every = 15), stop = nothing)
    out = String(take!(io))
    gs = r.history.constraint
    @test occursin("constraint: nonlinear (tol $(name === :OC ? "1e-04" : "1e-06"))", out)
    @test r.status === :maxiter && length(gs) == 30
    if name === :OC
      @test abs(gs[end]) <= 1e-4 && maximum(abs, gs[5:end]) <= 1e-3
    else
      @test all(<=(0), gs) && gs[end] >= -1e-6    # every accepted iterate feasible (the jump is restored at once)
    end
    @test all(isfinite, r.history.objective)
    # the physical volume in the fields at the end is the constrained one
    @test sum(∫(q.θ.ρ)q.dΩ) ≈ 0.4 * vol rtol = name === :OC ? 2e-4 : 2e-6
  end

  @testset "Checkpoint and resume with a nonlinear constraint" begin
    dir = mktempdir()
    vol = 1.0
    run(q, maxiter; kwargs...) = solve!(Optimization(q.z; objective = q.C, constraints = Volume(q.θ, q.dΩ; max_fraction = 0.4)),
                                        SIMP(OC()); maxiter, verbose = 0, stop = nothing, kwargs...)
    qa = compliance_problem()
    ra = run(qa, 8)
    qb = compliance_problem()
    run(qb, 4; checkpoint = Checkpoint(dir; every = 2))
    qc = compliance_problem()
    rc = run(qc, 8; resume = load_checkpoint(dir))
    @test rc.history.iteration == 1:8 && rc.history.objective[1:4] == ra.history.objective[1:4]
    @test abs(rc.history.constraint[end]) <= 1e-4
    # move limits and correction shift are saved: the resumed run is the uninterrupted one (up to threaded sums)
    @test rc.history.objective ≈ ra.history.objective rtol = 1e-10
  end

  @testset "Volume of the projected density: Taylor test (P1)" begin
    q = compliance_problem(; design = :P1)
    @test check_consistency(Volume(q.θ, q.dΩ; max_fraction = 0.4)).passed
  end

  @testset "Volume: infeasible start, linear volume of the design, errors" begin
    q = compliance_problem(; init = 0.9)
    vol = sum(∫(1.0)q.dΩ)
    V = Volume(q.θ, q.dΩ; max_fraction = 0.4)
    r = solve!(Optimization(q.z; objective = q.C, constraints = V), SIMP(SiMPL(); metric = LumpedMass(q.dΩ));
               maxiter = 3, verbose = 0, stop = nothing)
    @test r.start_scaling !== nothing && 0 < r.start_scaling < 1
    @test all(<=(0), r.history.constraint)
    # the volume of the design is linear: the same run as the hand-written linear functional
    runs = map((:volume, :hand)) do kind
      q = compliance_problem()
      C = kind === :volume ? Volume(q.x, q.dΩ; max_fraction = 0.4) : q.Vol
      solve!(Optimization(q.z; objective = q.C, constraints = C), SIMP(OC()); maxiter = 6, verbose = 0, stop = nothing)
    end
    @test runs[1].history.objective ≈ runs[2].history.objective rtol = 1e-12
    @test_throws ArgumentError Volume(q.θ, q.dΩ; max_fraction = -1.0)
    @test_throws ArgumentError Volume(q.θ, q.dΩ; max_fraction = 1.5)
    @test_throws ArgumentError Volume(q.θ, q.dΩ; max_fraction = 0.5, field = :E)
    @test_throws ArgumentError Optimization(q.z; objective = q.C, constraints = (V, V))
  end

  @testset "Stopping: range of the objective, kkt_rel, the grey monitor" begin
    h = OptHistory(30, ())
    rec!(f; kkt = NaN) = _record!(h, (h.n + 1, f, NaN, kkt, 0.1, 0.1, 1, 1, 0, 1, NaN, NaN, NaN), ())
    info(kkt = NaN) = HyperShape.Optimizers.StepInfo(true, 1.0, 1.0, kkt, 0.1, 0.1, 0.0, 1, NaN)
    st = StopCriteria(objective = 1e-3, window = 4)
    for f in (1.0, 1.01, 1.0, 1.01, 1.0)          # an oscillation: the ends of the window agree, the range does not
      rec!(f)
    end
    @test _stop_reason(st, info(), h, 0) === nothing
    @test _stop_reason(st, info(), h, 2) === nothing   # only 3 iterations of the stage
    for f in (0.9, 0.9004, 0.9002, 0.9001)
      rec!(f)
    end
    @test _stop_reason(st, info(), h, 0) === :objective
    # an objective tending to zero: floor 1e-3 |f₁|; a negative objective
    h0 = OptHistory(10, ())
    for f in (1.0, 1e-6, 0.8e-6, 0.6e-6, 0.5e-6)
      _record!(h0, (h0.n + 1, f, NaN, NaN, 0.1, 0.1, 1, 1, 0, 1, NaN, NaN, NaN), ())
    end
    @test _stop_reason(StopCriteria(objective = 1e-3, window = 4), info(), h0, 0) === :objective
    hn = OptHistory(10, ())
    for f in (-1.0, -2.0, -2.0005, -2.0004)
      _record!(hn, (hn.n + 1, f, NaN, NaN, 0.1, 0.1, 1, 1, 0, 1, NaN, NaN, NaN), ())
    end
    @test _stop_reason(StopCriteria(objective = 1e-3, window = 3), info(), hn, 0) === :objective
    @test _stop_reason(StopCriteria(objective = 1e-3, window = 4), info(), hn, 0) === nothing
    # kkt relative to the first iteration
    hk = OptHistory(10, ())
    # stationarity_rel: reference = first finite value (NaN rows from an old checkpoint are skipped)
    hs = OptHistory(10, ())
    for (k, sv) in enumerate((NaN, NaN, 1.0, 0.5, 0.05))
      _record!(hs, (k, 1.0, NaN, NaN, 0.1, 0.1, 1, 1, 0, 1, NaN, NaN, sv), ())
    end
    sinfo(S) = HyperShape.Optimizers.StepInfo(true, 1.0, 1.0, NaN, 0.1, 0.1, 0.0, 1, NaN, S)
    @test _stop_reason(StopCriteria(stationarity_rel = 0.1), sinfo(0.05), hs, 0) === :stationarity_rel
    @test _stop_reason(StopCriteria(stationarity_rel = 0.01), sinfo(0.05), hs, 0) === nothing
    @test _stop_reason(StopCriteria(stationarity = 0.1), sinfo(0.05), hs, 0) === :stationarity
    @test _stop_reason(StopCriteria(stationarity = 0.1), sinfo(0.05), hs, hs.n - 1) === nothing   # first of a stage
    hn2 = OptHistory(10, ())
    _record!(hn2, (1, 1.0, NaN, NaN, 0.1, 0.1, 1, 1, 0, 1, NaN, NaN, NaN), ())
    _record!(hn2, (2, 1.0, NaN, NaN, 0.1, 0.1, 1, 1, 0, 1, NaN, NaN, NaN), ())
    @test _stop_reason(StopCriteria(stationarity_rel = 0.1), sinfo(NaN), hn2, 0) === nothing
    _record!(hk, (1, 1.0, NaN, 10.0, 0.1, 0.1, 1, 1, 0, 1, NaN, NaN, NaN), ())
    _record!(hk, (2, 1.0, NaN, 0.05, 0.1, 0.1, 1, 1, 0, 1, NaN, NaN, NaN), ())
    @test _stop_reason(StopCriteria(kkt_rel = 1e-2), info(0.05), hk, 0) === :kkt_rel
    @test _stop_reason(StopCriteria(kkt_rel = 1e-2), info(0.05), hk, 1) === nothing   # first iteration of a stage
    @test _stop_reason(StopCriteria(kkt_rel = 1e-3), info(0.05), hk, 0) === nothing
    # the default criterion is reached on the small problem
    for (mk, lumped) in ((() -> SiMPL(), true), (() -> OC(), false))
      q = compliance_problem()
      method = lumped ? SIMP(mk(); metric = LumpedMass(q.dΩ)) : SIMP(mk())
      r = solve!(q.problem, method; maxiter = 300, verbose = 0)
      @test r.status === :converged && r.reason === :objective && r.iterations < 300
    end
    # grey monitor: history, table and closing line
    q = compliance_problem()
    io = IOBuffer()
    r = solve!(q.problem, SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 4, stop = nothing, io,
               grey = Grey(q.θ, q.dΩ))
    out = String(take!(io))
    ρ = q.θ.ρ
    @test r.history.grey[end] ≈ 4 * sum(∫(ρ * (1 - ρ))q.dΩ) / sum(∫(1.0)q.dΩ)
    @test all(g -> 0 <= g <= 1, r.history.grey)
    @test occursin("grey", out) && occursin(", grey ", out)
    @test all(isnan, solve!(compliance_problem().problem, SIMP(OC()); maxiter = 2, verbose = 0, stop = nothing).history.grey)
    @test_throws ArgumentError solve!(q.problem, SIMP(OC()); maxiter = 2, verbose = 0, grey = q.θ)
    θ2 = PhysicalFields(q.x; ρ = Identity(), E = :ρ => Identity())
    @test_throws ArgumentError Grey(θ2, q.dΩ)
    @test Grey(θ2, q.dΩ; field = :E).field === :E
    @test_throws MethodError Grey(q.x, q.dΩ)
    @test_throws ArgumentError Continuation(grey = q.H => [1.0, 2.0]; every = 2)   # a history column
    # grey through a checkpoint, P1 design
    dir = mktempdir()
    q1 = compliance_problem(; design = :P1)
    r1 = solve!(q1.problem, SIMP(OC()); maxiter = 3, verbose = 0, stop = nothing, grey = Grey(q1.θ, q1.dΩ),
                checkpoint = Checkpoint(dir; every = 3))
    @test all(isfinite, r1.history.grey) && load_checkpoint(dir).history.grey == r1.history.grey
    # banner: no "last stage only" without criteria
    io = IOBuffer()
    solve!(compliance_problem().problem, SIMP(OC()); maxiter = 1, io, verbose = 1, stop = StopCriteria(),
           schedule = Continuation(β = q.H => [1.0, 2.0]; every = 2))
    @test !occursin("last stage only", String(take!(io)))
  end

  @testset "Defaults: metric of OC, stopping criteria, failure handling and the banner" begin
    default_stop = HyperShape.Methods.default_stop
    @test SIMP(OC()).default_metric && !SIMP(OC(); metric = Euclidean()).default_metric
    @test RejectStep() === RejectStep(shrink = 0.5, max = 5)
    for o in (OC(), SiMPL())
      d = default_stop(o)
      @test d.objective == 1e-3 && d.window == 10 && d.kkt === nothing && d.kkt_rel === nothing && d.change === nothing
    end
    # the defaults give the same run as the same values written explicitly
    for (mkopt, lumped) in ((() -> OC(), false), (() -> SiMPL(), true))
      runs = map((true, false)) do dflt
        q = compliance_problem()
        method = lumped ? SIMP(mkopt(); metric = LumpedMass(q.dΩ)) : SIMP(mkopt())
        io = IOBuffer()
        r = dflt ? solve!(q.problem, method; maxiter = 8, io, verbose = 1) :
                   solve!(q.problem, method; maxiter = 8, io, verbose = 1, on_failure = RejectStep(shrink = 0.5, max = 5),
                          stop = default_stop(mkopt()))
        (r, String(take!(io)))
      end
      (r1, o1), (r2, o2) = runs
      @test r1.status == r2.status && r1.iterations == r2.iterations && r1.objective == r2.objective
      name = nameof(typeof(mkopt()))
      @test occursin("over 10 iterations (default)", o1) && !occursin("iterations (default)", o2)
      @test occursin("at most 5 attempts (default)", o1) && occursin("at most 5 attempts\n", o2)
    end
    io = IOBuffer()
    solve!(compliance_problem().problem, SIMP(OC()); maxiter = 2, io, verbose = 1, stop = nothing)
    out = String(take!(io))
    @test occursin("metric: Euclidean (default) (OC: only for the stationarity measure)", out) && occursin("stop: maxiter only · maxiter 2", out)
  end

  @testset "Screen output" begin
    io = IOBuffer()
    q = compliance_problem(; init = 0.6)
    cnt = Ref(0)
    r = solve!(q.problem, SIMP(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 6, io,
               stop = StopCriteria(kkt = 1e-12), schedule = Continuation(β = q.H => [1.0, 2.0]; every = 3),
               on_failure = RejectStep(shrink = 0.5, max = 3))
    out = String(take!(io))
    @test occursin("HyperShape · SIMP(SiMPL)", out) && occursin("continuation β: 1 → 2", out)
    @test occursin("scaled towards the lower bound", out)
    @test occursin("objective", out) && occursin("f/a/F", out)
    @test count(l -> occursin(r"^\s+\d+\s+\d\.\d{6}e[+-]\d\d", l), split(out, '\n')) == 6   # one row per iteration
    @test occursin("→ β = 2 (stage 2 of 2)", out)
    @test occursin("⚠ maximum number of iterations (6)", out) && occursin("solves: forward", out)
    # level 1: no rows
    io = IOBuffer()
    q = compliance_problem()
    solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 3, verbose = 1, io, stop = StopCriteria(change = 1.0),
           on_failure = RejectStep(shrink = 0.5, max = 3))
    out = String(take!(io))
    @test occursin("✓ converged", out) && !occursin("f/a/F", out)
    # a rejected step and a failure
    io = IOBuffer()
    cnt[] = 0
    q = compliance_problem(; solve_flag = () -> (cnt[] += 1; cnt[] != 3 && cnt[] < 6))
    @test_throws OptimizationFailure solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 6, io,
                                            on_failure = RejectStep(shrink = 0.5, max = 2), stop = nothing)
    out = String(take!(io))
    @test occursin("✗ iteration 2: the forward problem did not converge", out) && occursin("(2 attempts)", out)
    @test occursin("✗ failed after", out)
    @test_throws ArgumentError solve!(compliance_problem().problem, SIMP(OC(); metric = Euclidean()); maxiter = 2,
                                      verbose = 5, on_failure = RejectStep(shrink = 0.5, max = 3), stop = nothing)
  end
end
