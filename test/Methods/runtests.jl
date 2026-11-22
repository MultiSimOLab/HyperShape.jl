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

# an optimiser that does not compute the stationarity measure: SiMPL reporting NaN
struct NoStatOpt <: HyperShape.Optimizers.Optimizer end
const _NS = HyperShape.Optimizers
_NS.init(::NoStatOpt, z, oracle) = _NS.init(SiMPL(), z, oracle)
function _NS.step!(::NoStatOpt, state, z, oracle, ctx)
  i = _NS.step!(SiMPL(), state, z, oracle, ctx)
  _NS.StepInfo(i.accepted, i.f, i.f_new, i.kkt, i.change, i.step, _NS.violation(i), i.evaluations, i.multipliers[1], NaN)
end
_NS.shrink!(::NoStatOpt, state, f) = _NS.shrink!(SiMPL(), state, f)
_NS.reset!(::NoStatOpt, state) = _NS.reset!(SiMPL(), state)
_NS.constraint_tol(::NoStatOpt) = _NS.constraint_tol(SiMPL())
_NS.handles(::NoStatOpt, c::_NS.ConstraintInfo) = _NS.handles(SiMPL(), c)     # a third-party optimiser declares them
_NS.max_native(::NoStatOpt) = 1
# a method of the user: a projected gradient step with the volume as a native linear constraint (by bisection)
struct _GradStep <: HyperShape.Optimizers.Optimizer end
_NS.handles(::_GradStep, c::_NS.ConstraintInfo) = c.linear && c.hi == 0 && _NS.is_upper(c)
_NS.max_native(::_GradStep) = 1
_NS.init(::_GradStep, z, oracle) = (dF = similar(z), z0 = similar(z))
_NS.reset!(::_GradStep, s) = s
_NS.shrink!(::_GradStep, s, f) = s
_NS.computes_kkt(::_GradStep) = false
function _NS.step!(::_GradStep, s, z, oracle, ctx)
  f = _NS.objective(oracle, z)
  _NS.gradient!(s.dF, oracle, z)
  lb, ub = _NS.bounds(oracle)
  a, V = _NS.linear_constraint(oracle, 1)
  copyto!(s.z0, z)
  α = 0.05 / maximum(abs, s.dF)
  zc(λ) = clamp.(s.z0 .- α .* (s.dF .+ λ .* a), lb, ub)
  lo, hi = 0.0, 1e6
  for _ in 1:200
    mid = (lo + hi) / 2
    dot(a, zc(mid)) > V ? (lo = mid) : (hi = mid)
  end
  z .= zc(hi)
  _NS.StepInfo(true, f, NaN, NaN, maximum(abs, z .- s.z0), α, 0.0, 1, hi, NaN)
end
Base.@kwdef struct _GradMethod <: HyperShape.Methods.AbstractMethod
  bad::Bool = false
end
HyperShape.Methods.stepper(::_GradMethod) = _GradStep()
HyperShape.Methods.build_oracle(::_GradMethod, problem) = FunctionalOracle(problem, Euclidean())
HyperShape.Methods.augmented_lagrangian(::_GradMethod) = AugmentedLagrangian()
HyperShape.Methods.check_problem(m::_GradMethod, problem) = m.bad ? throw(ArgumentError("bad problem")) : nothing

# an optimiser of the user that does not declare handles/max_native
struct PlainOpt <: HyperShape.Optimizers.Optimizer end
_NS.init(::PlainOpt, z, oracle) = _NS.init(SiMPL(), z, oracle)
_NS.step!(::PlainOpt, state, z, oracle, ctx) = _NS.step!(SiMPL(), state, z, oracle, ctx)

# a monitor of the user
struct _CountMonitor
  n::Base.RefValue{Int}
end
HyperShape.monitor_value(m::_CountMonitor) = (m.n[] += 1; Float64(m.n[]))
struct _UserMetric
  w::Vector{Float64}
end
HyperShape.metric_weights(m::_UserMetric, z) = m.w
struct _StopAt
  n::Int
end
(s::_StopAt)(info, h, ctx) = h.n >= s.n
struct _TypedMetric
  w::Vector{Float64}
end
HyperShape.metric_weights(m::_TypedMetric, z::OptVariables) = m.w
struct _VecMonitor end
HyperShape.monitor_value(::_VecMonitor) = [1.0, 2.0]
struct _BadMonitor
  at::Int
  n::Base.RefValue{Int}
end
HyperShape.monitor_value(m::_BadMonitor) = (m.n[] += 1; m.n[] == m.at ? error("boom") : 1)   # an Int: converted


nonincreasing(v) = all(i -> v[i+1] <= v[i] + 1e-12 * abs(v[i]), 1:length(v)-1)

# MBB written with the two factories of Optimization(model; design_setup, physics_setup) (PR 3b): `proj` is created
# outside, `load` scales the load at each call of physics_setup (a stale capture on purpose when it changes)
function mbb_model(nx, ny)
  model  = CartesianDiscreteModel((0, nx, 0, ny), (nx, ny))
  labels = get_face_labeling(model)
  add_tag_from_vertex_filter!(labels, "sym", model, x -> x[1] ≈ 0)
  add_tag_from_vertex_filter!(labels, "roller", model, x -> x[1] ≈ nx && x[2] ≈ 0)
  add_tag_from_vertex_filter!(labels, "load", model, x -> x[2] ≈ ny && x[1] <= 1)
  add_tag_from_vertex_filter!(labels, "solid", model, x -> x[1] >= nx - 2 - 1e-9 && x[2] <= 2 + 1e-9)
  model
end

function mbb_factories(; order = 0, linear = false, al = false, frozen = false, proj = TanhProjection(β = 1.0, η = 0.5),
                         load = () -> 1.0, output = true, monitors = true)
  function design_setup(model)
    reffe = ReferenceFE(lagrangian, Float64, order)
    conf = order == 0 ? :L2 : :H1
    Vρ = frozen ? TestFESpace(model, reffe; conformity = conf, dirichlet_tags = "solid") : TestFESpace(model, reffe; conformity = conf)
    z = OptVariables(ρ = FieldVariable(Vρ; bounds = (0.0, 1.0), init = 0.5, frozen = frozen ? 1.0 : nothing))
    x = DesignFields(z; ρ = Identity())
    (; z, x)
  end
  function physics_setup(model, x)
    Ω, Γ   = Triangulation(model), BoundaryTriangulation(model; tags = "load")
    dΩ, dΓ = Measure(Ω, 2), Measure(Γ, 2)
    bc = BoundaryConditions(BoundaryCondition("sym", VectorValue(0.0, 0.0); mask = (true, false)),
                            BoundaryCondition("roller", VectorValue(0.0, 0.0); mask = (false, true)))
    V  = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1), bc; conformity = :H1)
    U  = TrialFESpace(V, bc, 1.0)
    filt = HelmholtzFilter(dΩ; rmin = 1.5, ls = LUSolver())
    θ  = PhysicalFields(x; ρ = linear ? filt : proj ∘ filt)
    ν, Emin = 0.3, 1e-9
    λ, μ = ν / (1 - ν^2), 1 / (2(1 + ν))
    _, _, ∂²Ψ, _, _, ∂²Ψ_∂ρ = EnergyInterpolationScheme(LinearElasticity2D(λ = λ, μ = μ),
                                                         LinearElasticity2D(λ = Emin * λ, μ = Emin * μ); p = 3.0)()
    F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
    c = load()
    t(x) = VectorValue(0.0, -c * (4 - 6x[1]))
    a(u, v) = ∫(∇(v)' ⊙ ((∂²Ψ ∘ (θ.ρ, F ∘ ∇(u)')) ⊙ ∇(u)'))dΩ
    S = StateOperator(StaticLinearModel(v -> ∫(t ⋅ v)dΓ, a, U, V, bc; ls = LUSolver(), backend = :compiled), θ;
          adjoint = AdjointProblem(sensitivity = (u, p) -> (δ -> ∫((∇(p)' ⊙ ((∂²Ψ_∂ρ ∘ (θ.ρ, F ∘ ∇(u)')) ⊙ ∇(u)')) * δ)dΩ)))
    C = Functional((u, θ) -> ∫(t ⋅ u)dΓ, S; dJdu = SelfAdjoint(), scale = :initial)
    Vol = Volume(θ, dΩ; max_fraction = 0.5)
    cons = al ? (volume = Vol, tight = Volume(θ, dΩ; max_fraction = 0.45)) : (volume = Vol,)
    r = (; objective = C, constraints = cons, θ, S, Ω, dΩ)
    monitors && (r = (; r..., monitors = (grey = Grey(θ, dΩ),)))
    output && (r = (; r..., output = (post, ctx) -> write_step(post, Ω, "rho" => θ.ρ, "u" => get_state(S))))
    r
  end
  design_setup, physics_setup
end

# a discretisation of the user: one rebuild at `at`, keeping the design or not
struct _KeepDesign <: HyperShape.Methods.Discretization
  at::Int
  what::Symbol
end
HyperShape.Methods.due(d::_KeepDesign, k, info, history, ctx) = (@assert history.n == k; k == d.at)
HyperShape.Methods.rediscretize(d::_KeepDesign, problem, state, ctx) =
  (@assert ctx.phase === :setup; HyperShape.Methods.rebuild(problem; what = d.what))

# every column of the history but the counts of solves
const _SAME_COLS = (:iteration, :objective, :kkt, :stationarity, :change, :step, :attempts)
same_history(a, b) = all(isequal(getproperty(a, c), getproperty(b, c)) for c in _SAME_COLS) &&
  isequal(a.constraints, b.constraints) && isequal(a.multipliers, b.multipliers) && isequal(a.violations, b.violations) &&
  isequal(a.penalties, b.penalties) && isequal(a.monitors, b.monitors) && isequal(a.continuation, b.continuation)

@testset "Methods" begin

  @testset "Several constraints: oracle and the check of the optimiser" begin
    q = compliance_problem()
    (; z, dΩ, θ) = q
    vol = sum(∫(1.0)dΩ)
    Vraw = Functional(θ -> sum(∫(θ.ρ)dΩ), q.θv; dJdθ = θ -> (δ -> ∫(δ)dΩ), scale = :bound, linear = true)
    B = q.boundary(scale = :initial)
    Vp = Functional(θ -> sum(∫(θ.ρ)dΩ), θ; dJdθ = θ -> (δ -> ∫(δ)dΩ), scale = 2.0)
    pb = Optimization(z; objective = q.C,
                      constraints = (volume = q.Vol, floor = Constraint(Vraw; min = 0.1vol), tip = Constraint(B; max = 1e3),
                                     band = Constraint(Vp; min = 0.1vol, max = 0.9vol)))
    o = FunctionalOracle(pb, Euclidean())
    @test HyperShape.Optimizers.num_constraints(o) == 4
    foreach(i -> HyperShape.Optimizers.linear_constraint(o, i) === nothing &&
                 HyperShape.Methods._is_linear(o, i) && _setup_constraint!(o, i, pb.ctx), 1:4)
    zk = o.zk
    copyto!(zk, get_values(z))
    infos = [HyperShape.Optimizers.constraint_info(o, i) for i in 1:4]
    @test infos[1].lo == -Inf && infos[1].hi == 0 && infos[1].linear && infos[1].design_only
    @test infos[2].hi == Inf && infos[2].lo == 0 && infos[2].linear          # g = (F − min)/s ≥ 0
    @test !infos[3].design_only && !infos[3].linear && infos[3].hi == 0
    @test infos[4].lo ≈ -0.8vol / 2.0 && infos[4].hi == 0 && !infos[4].linear
    f0 = HyperShape.Optimizers.objective(o, zk)
    gJ = HyperShape.Optimizers.gradient!(zeros(length(zk)), o, zk)
    gs = map(1:4) do i
      gi = HyperShape.Optimizers.constraint(o, i, zk)
      i == 2 && @test gi ≈ (sum(∫(q.θv.ρ)dΩ) - 0.1vol) / (0.1vol) rtol = 1e-12
      i == 4 && @test gi ≈ (sum(∫(θ.ρ)dΩ) - 0.9vol) / 2.0 rtol = 1e-12
      HyperShape.Optimizers.constraint_gradient!(zeros(length(zk)), o, i, zk)
    end
    HyperShape.Optimizers.objective(o, zk)
    w = [0.3, -1.2, 0.0, 2.0]
    dL = HyperShape.Optimizers.weighted_gradient!(zeros(length(zk)), o, 1.5, w, zk)
    @test dL ≈ 1.5 .* gJ .+ sum(w[i] .* gs[i] for i in 1:4) rtol = 1e-10
    @test_throws DimensionMismatch HyperShape.Optimizers.weighted_gradient!(zeros(length(zk)), o, 1.0, [1.0], zk)
    # constraints with weight zero (one on other physical fields): the gradient of the objective, bit for bit
    @test HyperShape.Optimizers.weighted_gradient!(zeros(length(zk)), o, 1.0, zeros(4), zk) == gJ
    # another active set: same tuple of functionals, nothing to compile
    w2 = [0.0, 0.0, 0.7, 0.0]
    dL2 = HyperShape.Optimizers.weighted_gradient!(zeros(length(zk)), o, 1.5, w2, zk)
    @test dL2 ≈ 1.5 .* gJ .+ 0.7 .* gs[3] rtol = 1e-10
    # a rejected trial: the current design served from the snapshot, then the weighted gradient there
    zt = clamp.(zk .+ 0.05 .* sin.(1:length(zk)), 0.0, 1.0)
    HyperShape.Optimizers.objective(o, zt)
    HyperShape.Optimizers.objective(o, zk)
    @test HyperShape.Optimizers.weighted_gradient!(zeros(length(zk)), o, 1.5, w, zk) ≈ dL rtol = 1e-10
    # who treats what: the volume native (OC, SiMPL), the others to the augmented Lagrangian, decided without solves
    split(opt, lag = AugmentedLagrangian()) = HyperShape.Methods._native_split(pb, opt, lag)
    @test split(OC()) == [1] && split(SiMPL()) == [1]
    @test split(SiMPL(), AugmentedLagrangian(native = ())) == Int[]
    @test_throws ArgumentError split(OC(), AugmentedLagrangian(native = ()))           # OC needs one of its own
    @test_throws ArgumentError split(SiMPL(), AugmentedLagrangian(native = (:tip,)))   # reads a state
    @test_throws ArgumentError split(SiMPL(), AugmentedLagrangian(native = (:nope,)))
    @test_throws ArgumentError split(SiMPL(), AugmentedLagrangian(native = (:volume, :volume)))
    lower = Optimization(z; objective = q.C, constraints = (floor = Constraint(Vraw; min = 0.1vol),))
    @test HyperShape.Methods._native_split(lower, SiMPL(), AugmentedLagrangian()) == Int[]
    @test_throws ArgumentError HyperShape.Methods._native_split(lower, OC(), AugmentedLagrangian())
    # the banner says who treats each constraint
    buf = IOBuffer()
    solve!(pb, DensityMethod(SiMPL(); metric = Euclidean()); maxiter = 1, verbose = 1, io = buf, stop = nothing)
    out = String(take!(buf))
    @test occursin("augmented Lagrangian: Bertsekas(μ0 = 1.0, growth = 1.0) (default)", out) && occursin("native", out)
    # OC and SiMPL check by themselves what they are given (an oracle used outside solve!)
    ol = FunctionalOracle(lower, Euclidean())
    _setup_constraint!(ol, 1, lower.ctx)
    @test_throws ArgumentError init(OC(), copy(get_values(z)), ol)
    @test_throws ArgumentError init(SiMPL(), copy(get_values(z)), ol)
    # an optimiser of the user that does not declare what it treats: every constraint goes to the augmented Lagrangian
    @test HyperShape.Methods._native_split(q.problem, PlainOpt(), AugmentedLagrangian()) == Int[]
    # a constraint that cannot be evaluated at the start, with scale = :initial
    Vnan = Functional(θ -> NaN * sum(∫(θ.ρ)dΩ), q.θv; dJdθ = θ -> (δ -> ∫(δ)dΩ), scale = :initial)
    on = FunctionalOracle(Optimization(z; objective = q.C, constraints = (bad = Vnan,)), Euclidean())
    err = try HyperShape.Optimizers.constraint_info(on, 1); nothing catch e; e end
    @test err isa ArgumentError && occursin("not finite", err.msg)
  end

  @testset "Augmented Lagrangian: two constraints with the state (thermal)" begin
    # minimum physical volume s.t. compliance ≤ 1.3 C₀ (self-adjoint) and mean boundary value ≤ 1.2 B₀ (own adjoint)
    p = state_problem(; n = 16, init = fill(0.6, 256))
    (; θ, dΩ) = p
    C0 = functional_value(p.compliance(scale = 1.0), nothing)
    B0 = functional_value(p.boundary(scale = 1.0), nothing)
    vol = Functional(θ -> ∫(θ.ρ)dΩ, θ; dJdθ = θ -> (δ -> ∫(δ)dΩ), scale = 1.0)
    pb = Optimization(objective = vol, constraints = (compliance = Constraint(p.compliance(scale = :bound); max = 1.3C0),
                                                      boundary = Constraint(p.boundary(scale = :bound); max = 1.2B0)))
    r = solve!(pb, DensityMethod(SiMPL(); metric = LumpedMass(dΩ)); maxiter = 200, verbose = 0, stop = nothing)
    h = r.history
    # with a fixed penalty the step oscillates around the active constraint, and the trajectory depends on the rounding
    # of the machine (CPU, --check-bounds): look at the last 50 iterations, with margin over what CI and the Mac give
    # (largest violation 3e-3 to 8e-3, mean below 1e-3)
    vb = h.violations.boundary[end-49:end]
    @test maximum(h.violations.compliance[end-49:end]) == 0
    @test minimum(vb) <= 1e-4 && sum(vb) / length(vb) <= 2e-3 && maximum(vb) <= 2e-2
    @test h.multipliers.compliance[end] == 0 && h.constraints.compliance[end] < -0.1           # inactive
    @test h.multipliers.boundary[end] > 0 && minimum(abs, h.constraints.boundary[end-9:end]) < 1e-3   # active
    @test h.penalties.compliance[end] == 1.0 && h.penalties.boundary[end] == 1.0
    @test h.objective[end] < 0.6                         # below the uniform start
    @test sum(h.adjoint) <= r.iterations                  # at most one adjoint solve per iteration (one sweep)
    # LinearInState: the same derivative as the form written by hand
    Bl = Functional((u, θ) -> ∫(u)p.dΓ, p.S; dJdu = LinearInState(), fields = θ, scale = 1.0)
    g1 = functional_gradient!(zeros(length(p.z)), Bl, nothing)
    g2 = functional_gradient!(zeros(length(p.z)), p.boundary(scale = 1.0), nothing)
    @test g1 ≈ g2 rtol = 1e-12
  end

  @testset "Augmented Lagrangian with a native constraint, a lower bound and a checkpoint" begin
    # minimum compliance s.t. physical volume ≤ 0.4 (native) and mean boundary value ≥ 0.5 B₀ (augmented Lagrangian)
    mk() = begin
      p = state_problem(; n = 16, init = fill(0.4, 256))
      B0 = functional_value(p.boundary(scale = 1.0), nothing)
      pb = Optimization(objective = p.compliance(scale = :initial),
                        constraints = (volume = Volume(p.θ, p.dΩ; max_fraction = 0.4),
                                       boundary = Constraint(p.boundary(scale = :bound); min = 0.5B0)))
      p, pb
    end
    p, pb = mk()
    r = solve!(pb, DensityMethod(SiMPL(); metric = LumpedMass(p.dΩ)); maxiter = 320, verbose = 0, stop = nothing)
    h = r.history
    @test h.violations.volume[end] <= 1e-6 && h.violations.boundary[end] <= 1e-4
    @test h.multipliers.boundary[end] < 0                 # the lower bound is active
    @test isnan(h.penalties.volume[end]) && h.penalties.boundary[end] == 1.0
    # a resumed run continues as the uninterrupted one (OC saves its memory; λ and μ are in the checkpoint)
    dir = mktempdir()
    runs = map((nothing, dir)) do resume
      p, pb = mk()
      resume === nothing ||
        solve!(pb, DensityMethod(OC()); maxiter = 6, verbose = 0, stop = nothing, checkpoint = Checkpoint(filepath = dir, every = 6))
      p, pb = mk()
      solve!(pb, DensityMethod(OC()); maxiter = 12, verbose = 0, stop = nothing, resume)
    end
    @test runs[1].history.objective == runs[2].history.objective
    @test runs[1].history.multipliers.boundary == runs[2].history.multipliers.boundary
    @test get_values(runs[1].z) == get_values(runs[2].z)
    ck = load_checkpoint(dir)
    @test ck.handled == Dict(:volume => :native, :boundary => :lagrangian) && ck.lagrangian.updates == 6
    # the split of the checkpoint must be the one of the run (both ways)
    p, pb = mk()
    pc = mktempdir()
    solve!(pb, DensityMethod(SiMPL(); metric = LumpedMass(p.dΩ), lagrangian = AugmentedLagrangian(native = ()));
           maxiter = 2, verbose = 0, stop = nothing, checkpoint = Checkpoint(filepath = pc, every = 2))
    p, pb = mk()
    @test_throws ArgumentError solve!(pb, DensityMethod(SiMPL(); metric = LumpedMass(p.dΩ)); maxiter = 4, verbose = 0, resume = pc)
    # a monitor that throws: the checkpoint has the update of that iteration
    p, pb = mk()
    pf = mktempdir()
    @test_throws OptimizationFailure solve!(pb, DensityMethod(OC()); maxiter = 6, verbose = 0, stop = nothing,
                                            monitors = (b = _BadMonitor(3, Ref(0)),), checkpoint = Checkpoint(filepath = pf, every = 10))
    @test load_checkpoint(pf).lagrangian.updates == load_checkpoint(pf).iteration == 3
    # a run that ends with a constraint violated says so
    p, _ = mk()
    pv = Optimization(objective = p.compliance(scale = :initial),
                      constraints = (far = Constraint(p.boundary(scale = :bound); min = 3 * functional_value(p.boundary(scale = 1.0), nothing)),))
    buf = IOBuffer()
    solve!(pv, DensityMethod(SiMPL(); metric = LumpedMass(p.dΩ)); maxiter = 1, verbose = 1, io = buf, stop = nothing)
    @test occursin("constraints not met to their tolerance", String(take!(buf)))
    p, pb = mk()
    @test_throws ArgumentError solve!(pb, DensityMethod(SiMPL(); metric = LumpedMass(p.dΩ),
                                      lagrangian = AugmentedLagrangian(native = ())); maxiter = 12, verbose = 0, resume = dir)
    # a checkpoint of another version says which
    tfile = joinpath(dir, "checkpoint.toml")
    write(tfile, replace(read(tfile, String), "hypershape_checkpoint = 3" => "hypershape_checkpoint = 1"))
    err = try load_checkpoint(dir); nothing catch e; e end
    @test err isa ArgumentError && occursin("version 1", err.msg)
  end

  @testset "Frozen zones in the loop: consistency, banner, checkpoint" begin
    for order in (0, 1)
      q = frozen_mbb(; order)
      rep = check_consistency((compliance = q.C, volume = q.Vol))
      @test rep.passed && any(r -> occursin("frozen zone of θ.ρ", r.label) && occursin("8 cells", r.message), rep.results)
    end
    q = frozen_mbb()
    buf = IOBuffer()
    r = solve!(q.problem, DensityMethod(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 20, verbose = 1, io = buf, stop = nothing)
    @test occursin("ρ (88, 8 frozen cells)", String(take!(buf)))
    χ = frozen_indicator(q.x, :ρ)
    @test sum(∫(χ * (q.θ.ρ - 1.0) * (q.θ.ρ - 1.0))q.dΩ) == 0 && r.history.constraints.volume[end] <= 1e-6
    # a resumed run continues as the uninterrupted one (OC), with the frozen cells in the checkpoint
    dir = mktempdir()
    runs = map((nothing, dir)) do resume
      q = frozen_mbb()
      resume === nothing ||
        solve!(q.problem, DensityMethod(OC()); maxiter = 4, verbose = 0, stop = nothing, checkpoint = Checkpoint(filepath = dir, every = 4))
      q = frozen_mbb()
      solve!(q.problem, DensityMethod(OC()); maxiter = 8, verbose = 0, stop = nothing, resume)
    end
    @test runs[1].history.objective == runs[2].history.objective && get_values(runs[1].z) == get_values(runs[2].z)
    @test load_checkpoint(dir).frozen_cells == Dict(:ρ => 8)
  end

  @testset "Constraint(F; max = 0) is the functional alone (bit for bit)" begin
    hs = map((c -> c, c -> Constraint(c; max = 0.0))) do wrap
      q = compliance_problem()
      pb = Optimization(q.z; objective = q.C, constraints = (volume = wrap(q.Vol),))
      r = solve!(pb, DensityMethod(OC(); metric = Euclidean()); maxiter = 6, verbose = 0, stop = nothing)
      (r.history.objective, r.history.constraints.volume, copy(get_values(r.z)))
    end
    @test hs[1] == hs[2]
  end

  @testset "Oracle: volume vector, linearity, metric ($(design))" for design in (:P0, :P1)
    q = compliance_problem(; design)
    m = collect(assemble_vector(v -> ∫(v)q.dΩ, q.Vρ))
    o = FunctionalOracle(q.problem, LumpedMass(q.dΩ))
    @test o.metric ≈ m
    design === :P0 && @test all(≈(1 / 64), o.metric)
    _setup_constraint!(o, 1, q.problem.ctx)
    @test o.a[1] ≈ m ./ (0.4 * sum(m))
    @test o.V[1] ≈ 1.0 atol = 1e-12            # (∫ρ - 0.4|Ω|)/(0.4|Ω|) ≤ 0  ⇔  aᵀz ≤ 1
    # a projected volume is not linear in z
    Vp = Functional(θ -> sum(∫(θ.ρ)q.dΩ) - 0.4, q.θ; dJdθ = θ -> (δ -> ∫(δ)q.dΩ), scale = 0.4, linear = true)
    bad = Optimization(q.z; objective = q.C, constraints = (volume = Vp,))
    @test_throws ArgumentError solve!(bad, DensityMethod(OC(); metric = Euclidean()); maxiter = 2, verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
    @test FunctionalOracle(q.problem, Euclidean()).metric === nothing
    # a uniform start at 0.3 used to hide a nonlinear constraint
    q3 = compliance_problem(; design, init = 0.3)
    Vp3 = Functional(θ -> sum(∫(θ.ρ)q3.dΩ) - 0.4, q3.θ; dJdθ = θ -> (δ -> ∫(δ)q3.dΩ), scale = 0.4, linear = true)
    @test_throws ArgumentError solve!(Optimization(q3.z; objective = q3.C, constraints = (volume = Vp3,)),
                                      DensityMethod(OC(); metric = Euclidean()); maxiter = 2, verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
    @test get_values(q3.z) == fill(0.3, length(q3.z))     # z restored after the probes
    # the filtered, unprojected volume is linear and has the same vector (Neumann filter)
    θf = PhysicalFields(q.x; ρ = HelmholtzFilter(q.Sf, q.dΩ))
    Vf = Functional(θ -> sum(∫(θ.ρ)q.dΩ) - 0.4 * sum(m), θf; dJdθ = θ -> (δ -> ∫(δ)q.dΩ), scale = 0.4 * sum(m), linear = true)
    of = FunctionalOracle(Optimization(q.z; objective = q.C, constraints = (volume = Vf,)), Euclidean())
    _setup_constraint!(of, 1, q.problem.ctx)
    @test of.a[1] ≈ m ./ (0.4 * sum(m)) rtol = 1e-10
  end

  @testset "Problem and method arguments" begin
    q = compliance_problem()
    c = Optimization(q.z; objective = q.C, constraints = (volume = q.Vol,)).constraints
    @test keys(c) == (:volume,) && c.volume.F === q.Vol && (c.volume.lo, c.volume.hi) == (-Inf, 0.0)
    @test Optimization(objective = q.C, constraints = (volume = q.Vol,)).z === q.z          # z deduced
    @test_throws ArgumentError Optimization(q.z; objective = q.C, constraints = q.Vol)       # not named
    @test_throws ArgumentError Optimization(q.z; objective = q.C, constraints = (q.Vol,))
    # several constraints and constraints with states are problems; whether the optimiser treats them, solve! checks
    @test length(Optimization(q.z; objective = q.C, constraints = (a = q.Vol, b = q.Vol)).constraints) == 2
    @test Optimization(q.z; objective = q.C, constraints = (c = q.C,)).constraints.c.F === q.C
    @test_throws ArgumentError Optimization(q.z; objective = q.C, constraints = (c = 1.0,))   # not a functional
    @test_throws ArgumentError Optimization(q.z; objective = q.C, constraints = (objective = q.Vol,))
    @test occursin("volume", sprint(show, Optimization(objective = q.C, constraints = (volume = q.Vol,))))
    # short show methods (no fields of the mesh in the REPL)
    for obj in (q.problem, q.C, q.S, q.θ, q.x, q.z, q.Sf)
      @test length(sprint(show, obj)) < 200
    end
    # a monitor of the user: any object with a monitor_value method
    qm = compliance_problem()
    rm_ = solve!(qm.problem, DensityMethod(OC()); maxiter = 3, verbose = 0, stop = nothing,
                 monitors = (count = _CountMonitor(Ref(0)), grey = Grey(qm.θ, qm.dΩ)))
    @test rm_.history.monitors.count == [1.0, 2.0, 3.0] && keys(rm_.history.monitors) == (:count, :grey)
    @test rm_.z === qm.z && rm_.time > 0 && occursin("OptResult(:maxiter", sprint(show, rm_))
    @test_throws ArgumentError solve!(qm.problem, DensityMethod(OC()); maxiter = 2, verbose = 0, monitors = (bad = 1.0,))
    @test_throws ArgumentError solve!(qm.problem, DensityMethod(OC()); maxiter = 2, verbose = 0, monitors = Grey(qm.θ, qm.dΩ))
    @test rm_.history.monitors.count[end] == 3.0                  # evaluated once per iteration, not at the close
    for bad in (:objective, :kkt, :S, :μ, :constraint)
      @test_throws ArgumentError solve!(qm.problem, DensityMethod(OC()); maxiter = 2, verbose = 0,
                                        monitors = NamedTuple{(bad,)}((Grey(qm.θ, qm.dΩ),)))
    end
    @test_throws ArgumentError solve!(qm.problem, DensityMethod(OC()); maxiter = 2, verbose = 0,
                                      continuation = Continuation(β = qm.H => [1.0]; every = 2),
                                      monitors = (β = Grey(qm.θ, qm.dΩ),))
    # a monitor that throws: OptimizationFailure with the history and the checkpoint of that iteration
    qb = compliance_problem()
    dirb = mktempdir()
    e = try
      solve!(qb.problem, DensityMethod(OC()); maxiter = 6, verbose = 0, stop = nothing, monitors = (b = _BadMonitor(3, Ref(0)),),
             checkpoint = Checkpoint(filepath = dirb, every = 10))
    catch err
      err
    end
    @test e isa OptimizationFailure && occursin("monitor threw at iteration 3", e.msg) && e.iterations == 3
    @test e.history.monitors.b[1:2] == [1.0, 1.0] && isnan(e.history.monitors.b[3])
    @test load_checkpoint(dirb).iteration == 3
    # any other exception (here, the post driver) writes a checkpoint at the last accepted design and is rethrown
    qp = compliance_problem()
    dirp = mktempdir()
    postp = OptPostProcessor((p, ctx) -> ctx.iteration == 3 && error("driver"); every = 1, initial = false)
    @test_throws ErrorException solve!(qp.problem, DensityMethod(OC()); maxiter = 6, verbose = 0, stop = nothing,
                                       checkpoint = Checkpoint(filepath = dirp, every = 10), post = postp,
                                       monitors = (grey = Grey(qp.θ, qp.dΩ),))
    ckp = load_checkpoint(dirp)
    @test ckp.iteration == 3 && ckp.history.n == 3 && length(ckp.history.monitors.grey) == 3
    # a monitor that returns something else than a number
    qn = compliance_problem()
    @test_throws OptimizationFailure solve!(qn.problem, DensityMethod(OC()); maxiter = 2, verbose = 0, stop = nothing,
                                            monitors = (v = _VecMonitor(),))
    # wrong resume, positional constructors validated, public names
    @test_throws ArgumentError solve!(qm.problem, DensityMethod(OC()); maxiter = 2, verbose = 0, resume = Checkpoint(filepath = dirb, every = 1))
    @test_throws ArgumentError StopCriteria(-1.0, 10, nothing, nothing, nothing, nothing, nothing)
    @test_throws ArgumentError RejectStep(2.0, 1)
    @test_throws ArgumentError Checkpoint("x", 0)
    for n in (:get_bounds, :changed!, :set_beta!, :design_gradient!, :update_fields!, :is_linear_field, :state_version,
              :invalidate!, :refresh!, :default_stop, :default_metric, :monitor_value)
      @test Base.ispublic(HyperShape, n) && !Base.isexported(HyperShape, n) && isdefined(HyperShape, n)
    end
    for obj in (Grey(qm.θ, qm.dΩ), Continuation(β = qm.H => [1.0, 2.0]; every = 2), DensityMethod(OC()), qm.H,
                TanhProjection(β = 2.0, η = 0.5) ∘ HelmholtzFilter(qm.Sf, qm.dΩ), Checkpoint(filepath = "x", every = 2))
      @test length(sprint(show, obj)) < 120
    end
    @test DensityMethod(OC()).metric isa Euclidean           # default metric of OC (which does not use it)
    @test_throws ArgumentError DensityMethod(SiMPL())         # SiMPL needs one
    @test_throws ArgumentError DensityMethod(OC(); metric = :lumped)
    @test_throws UndefKeywordError solve!(q.problem, DensityMethod(OC()))   # maxiter is mandatory
    @test_throws ArgumentError solve!(q.problem, DensityMethod(OC()); maxiter = 2, on_failure = :reject)
    @test_throws ArgumentError solve!(q.problem, DensityMethod(OC()); maxiter = 2, on_failure = nothing)
    @test_throws ArgumentError RejectStep(shrink = 1.5, max_attempts = 2)
    @test_throws ArgumentError StopCriteria(change_tol = -1)
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
      r = solve!(q.problem, DensityMethod(opt; metric); maxiter = 15, verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3),
                 post = OptPostProcessor((_, ctx) -> push!(seen, (ctx.iteration, ctx.attempt, ctx.phase)); every = 1, initial = false), stop = nothing)
      h = r.history
      @test r.status === :maxiter && r.iterations == 15 && length(h) == 15
      @test h.objective[end] < 0.9 * h.objective[1]
      @test all(<=(2e-3), h.constraints.volume)                      # OC: up to its bisection tolerance
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
    (; z, θ, S, dΓ) = q
    B = Functional((u, θ) -> ∫(u)dΓ, S; dJdu = u -> (w -> ∫(w)dΓ), fields = θ, scale = :initial)
    r = solve!(Optimization(z; objective = B, constraints = (volume = q.Vol,)), DensityMethod(OC(); metric = Euclidean());
               maxiter = 6, verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
    @test all(==(1), r.history.adjoint[2:end])
    @test r.history.objective[end] < r.history.objective[1]
  end

  @testset "Rejected steps" begin
    # the third forward solve fails once
    cnt = Ref(0)
    q = compliance_problem(; solve_flag = () -> (cnt[] += 1; cnt[] != 3))
    # scalar move limit (adapt = nothing): exact values of the step; the adaptive case is tested in Optimizers
    r = solve!(q.problem, DensityMethod(OC(adapt = nothing); metric = Euclidean()); maxiter = 6, verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
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
    @test_throws OptimizationFailure solve!(q.problem, DensityMethod(OC(); metric = Euclidean()); maxiter = 6,
                                            verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 2),
                                            post = OptPostProcessor((_, ctx) -> push!(accepted, copy(get_values(q.z))); every = 1, initial = false), stop = nothing)
    @test get_values(q.z) == last(accepted)
    # the forward problem fails at the start
    q = compliance_problem(; solve_flag = () -> false)
    err = try
      solve!(q.problem, DensityMethod(SiMPL(); metric = Euclidean()); maxiter = 3, verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 2), stop = nothing)
      nothing
    catch e
      e
    end
    @test err isa OptimizationFailure && occursin("first iteration", sprint(showerror, err))
    @test err.iterations == 0 && isempty(err.history.iteration) && all(isempty, err.history.continuation)
  end

  @testset "OC: move limit per continuation stage, banner" begin
    for adapt in (nothing, (0.7, 1.2))
      q = compliance_problem()
      oc = OC(; adapt)
      io = IOBuffer()
      r = solve!(q.problem, DensityMethod(oc); maxiter = 6, stop = nothing, io, verbose = 1,
                 continuation = Continuation(β = q.H => [1.0, 2.0, 4.0], move = oc => [0.2, 0.1, 0.05]; every = 2))
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
    @test_throws ArgumentError solve!(q.problem, DensityMethod(OC()); maxiter = 2, verbose = 0,
                                      continuation = Continuation(move = OC() => [0.2, 0.1]; every = 1))
    # the banner shows the move limit of the first stage
    oc = OC()
    io = IOBuffer()
    solve!(q.problem, DensityMethod(oc); maxiter = 1, stop = nothing, io, verbose = 1,
           continuation = Continuation(move = oc => [0.1, 0.05]; every = 2))
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
    r = solve!(q.problem, DensityMethod(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 12, continuation = sched,
               verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
    @test r.history.continuation.β[1:9] == [1, 1, 1, 2, 2, 2, 4, 4, 4]
    @test q.H.β == 4.0 && keys(r.history.continuation) == (:β,)
    # a criterion always met: one iteration per stage, stop in the last one
    q = compliance_problem()
    r = solve!(q.problem, DensityMethod(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 12, stop = StopCriteria(change_tol = 1.0),
               continuation = Continuation(β = q.H => [1.0, 2.0, 4.0]; every = 3, close_early = true), verbose = 0,
               on_failure = RejectStep(shrink = 0.5, max_attempts = 3))
    @test r.status === :converged && r.reason === :change_tol && r.iterations == 6   # not at the first iteration of a stage
    @test r.history.continuation.β == [1, 1, 2, 2, 4, 4]
    # by default the criteria only act in the last stage: the earlier stages run their `every` iterations
    q = compliance_problem()
    r = solve!(q.problem, DensityMethod(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 12, stop = StopCriteria(change_tol = 1.0),
               continuation = Continuation(β = q.H => [1.0, 2.0, 4.0]; every = 3), verbose = 0)
    @test r.status === :converged && r.iterations == 8 && r.history.continuation.β == [1, 1, 1, 2, 2, 2, 4, 4]
    # kkt with an optimiser that does not compute it
    q = compliance_problem()
    @test_throws ArgumentError solve!(q.problem, DensityMethod(OC(); metric = Euclidean()); maxiter = 3, stop = StopCriteria(kkt_tol = 1e-3),
                                      verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3))
    q = compliance_problem()
    r = solve!(q.problem, DensityMethod(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 200, stop = StopCriteria(objective_range = 1e-4, window = 3),
               verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3))
    @test r.status === :converged && r.reason === :objective_range && r.iterations < 200
    # stationarity measure: recorded by SiMPL and OC, absolute and relative criteria
    for opt in (SiMPL(), OC())
      q = compliance_problem()
      r = solve!(q.problem, DensityMethod(opt; metric = LumpedMass(q.dΩ)); maxiter = 8, stop = nothing, verbose = 0,
                 on_failure = RejectStep(shrink = 0.5, max_attempts = 3))
      S = r.history.stationarity
      @test length(S) == r.iterations && all(isfinite, S) && all(>(0), S)
      q = compliance_problem()
      r2 = solve!(q.problem, DensityMethod(opt; metric = LumpedMass(q.dΩ)); maxiter = 50, verbose = 0,
                  stop = StopCriteria(stationarity_tol = 2 * minimum(S)), on_failure = RejectStep(shrink = 0.5, max_attempts = 3))
      @test r2.status === :converged && r2.reason === :stationarity_tol && r2.iterations <= 8
      q = compliance_problem()
      r3 = solve!(q.problem, DensityMethod(opt; metric = LumpedMass(q.dΩ)); maxiter = 50, verbose = 0,
                  stop = StopCriteria(stationarity_rtol = 2 * minimum(S) / S[1]), on_failure = RejectStep(shrink = 0.5, max_attempts = 3))
      @test r3.status === :converged && r3.reason === :stationarity_rtol && r3.iterations <= 8
    end
    @test_throws ArgumentError StopCriteria(stationarity_tol = -1.0)
    @test_throws ArgumentError StopCriteria(stationarity_rtol = -1.0)
    # OC and SiMPL give the same S at the same design and metric
    q1, q2 = compliance_problem(), compliance_problem()
    r1 = solve!(q1.problem, DensityMethod(OC(); metric = LumpedMass(q1.dΩ)); maxiter = 1, stop = nothing, verbose = 0)
    r2 = solve!(q2.problem, DensityMethod(SiMPL(); metric = LumpedMass(q2.dΩ)); maxiter = 1, stop = nothing, verbose = 0)
    @test r1.history.stationarity[1] ≈ r2.history.stationarity[1] rtol = 1e-10
    # an optimiser that does not compute it (stationarity NaN): error when a criterion asks for it
    q = compliance_problem()
    @test_throws ArgumentError solve!(q.problem, DensityMethod(NoStatOpt(); metric = LumpedMass(q.dΩ)); maxiter = 3, verbose = 0,
                                      stop = StopCriteria(stationarity_tol = 1e-3))
    q = compliance_problem()
    rn = solve!(q.problem, DensityMethod(NoStatOpt(); metric = LumpedMass(q.dΩ)); maxiter = 2, verbose = 0, stop = nothing)
    @test all(isnan, rn.history.stationarity)
  end

  @testset "Checkpoint and resume" begin
    dir = mktempdir()
    run(q, maxiter; kwargs...) = solve!(q.problem, DensityMethod(OC(); metric = Euclidean()); maxiter, verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), kwargs..., stop = nothing)
    qa = compliance_problem()
    ra = run(qa, 8)
    qb = compliance_problem()
    rb = run(qb, 4; checkpoint = Checkpoint(filepath = dir, every = 2))
    @test isfile(joinpath(dir, "checkpoint.toml")) && isfile(joinpath(dir, "z_4.bin")) && isfile(joinpath(dir, "history_4.csv"))
    @test !isfile(joinpath(dir, "z_2.bin")) && !isfile(joinpath(dir, "memory_2.bin"))   # older files deleted
    ck = load_checkpoint(dir)
    @test ck.optimizer == "OC" && keys(ck.memory) == (:started, :shift, :m, :dprev, :z_old)   # adaptive move limits saved
    @test ck isa OptState && ck.method == "DensityMethod(OC)" && haskey(ck.linear_offsets, :volume)
    @test ck.iteration == 4 && ck.z == get_values(qb.z) && length(ck.history.objective) == 4
    qc = compliance_problem()
    rc = run(qc, 8; resume = ck)
    @test get_values(qc.z) == get_values(qa.z)            # OC with adaptive limits resumes exactly
    @test rc.history.objective == ra.history.objective && rc.history.iteration == 1:8
    @test qc.C.scale[] == qa.C.scale[]                    # the scale fixed in the first run
    # SiMPL resumes (not bit-identical) and the checks on the checkpoint
    qd = compliance_problem()
    rd = solve!(qd.problem, DensityMethod(SiMPL(); metric = LumpedMass(qd.dΩ)); maxiter = 6, resume = ck,
                verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
    @test rd.iterations == 6 && rd.history.objective[end] <= rd.history.objective[5] * (1 + 1e-12)
    qe = compliance_problem(; design = :P1)
    @test_throws ArgumentError run(qe, 8; resume = ck)
    @test_throws ArgumentError run(compliance_problem(), 4; resume = ck)   # already at maxiter
  end

  @testset "SiMPL resumes exactly; problem.state; LumpedMass(); multipliers of the result" begin
    hcols(h) = (h.objective, h.kkt, h.stationarity, h.change, h.step, h.constraints.volume, h.multipliers.volume)
    for (design, kind) in ((:P0, :linear), (:P1, :projected))
      mk() = (q = compliance_problem(; design);
              V = kind === :linear ? q.Vol : Volume(q.θ, q.dΩ; max_fraction = 0.4);
              (q, Optimization(q.z; objective = q.C, constraints = (volume = V,))))
      run(q, pb, maxiter; kw...) = solve!(pb, DensityMethod(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter, verbose = 0, stop = nothing, kw...)
      qa, pa = mk(); ra = run(qa, pa, 10)
      dir = mktempdir()
      qb, pb = mk(); run(qb, pb, 5; checkpoint = Checkpoint(filepath = dir, every = 5))
      ck = load_checkpoint(dir)
      @test keys(ck.memory) == (:k, :μ, :α_prev, :shift, :ψ, :ψ_prev, :z_prev, :g_prev)
      qc, pc = mk(); rc = run(qc, pc, 10; resume = ck)
      @test hcols(rc.history) == hcols(ra.history) && get_values(qc.z) == get_values(qa.z)
      # without files: the state the problem keeps
      @test pb.state[] isa OptState && pb.state[].iteration == 5
      qd, pd = mk(); rd = run(qd, pd, 10; resume = pb.state[])
      @test hcols(rd.history) == hcols(ra.history)
      @test ra.multipliers.volume == ra.history.multipliers.volume[end]
    end
    q = compliance_problem(; design = :P1)
    @test HyperShape.Methods._metric_vector(LumpedMass(), q.z) ≈ HyperShape.Methods._metric_vector(LumpedMass(q.dΩ), q.z) rtol = 1e-14
  end

  @testset "A method of the user runs in the common loop" begin
    q = compliance_problem()
    r = solve!(q.problem, _GradMethod(); maxiter = 15, verbose = 0, stop = nothing)
    @test r.iterations == 15 && r.history.objective[end] < r.history.objective[1]
    @test all(v -> v <= 1e-8, r.history.violations.volume)          # the volume in the augmented Lagrangian... or met
    buf = IOBuffer()
    solve!(compliance_problem().problem, _GradMethod(); maxiter = 1, verbose = 1, io = buf, stop = nothing)
    @test occursin("_GradMethod", String(take!(buf)))
    @test_throws ArgumentError solve!(compliance_problem().problem, _GradMethod(bad = true); maxiter = 1, verbose = 0)
  end

  @testset "3D ($(design))" for design in (:P0, :P1)
    q = compliance_problem(; design, dim = 3, n = 4)
    @test check_consistency(q.C, q.Vol).passed
    r = solve!(q.problem, DensityMethod(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 5, verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
    @test r.history.objective[end] < r.history.objective[1] && nonincreasing(r.history.objective)
  end

  @testset "Stalled: z, fields and state at the last accepted design" begin
    sign = Ref(1.0)
    q = compliance_problem(; sign)
    r = solve!(q.problem, DensityMethod(SiMPL(max_backtracks = 1); metric = LumpedMass(q.dΩ)); maxiter = 10,
               verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), post = OptPostProcessor((_, ctx) -> (ctx.iteration == 2 && (sign[] = -1.0)); every = 1, initial = false), stop = nothing)
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
      solve!(q.problem, DensityMethod(OC(); metric = Euclidean()); maxiter = 6, verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 2),
             checkpoint = Checkpoint(filepath = dir, every = 10), post = OptPostProcessor((_, ctx) -> push!(accepted, copy(get_values(q.z))); every = 1, initial = false), stop = nothing)
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
    (; z, θ, S, dΓ, model_p, sens) = q
    # an adjoint solve that gives a non-finite solution
    Sbad = StateOperator(S.model, θ; adjoint = AdjointProblem(model_p; sensitivity = sens,
                                                              solve = (m, b, ctx, stale) -> fill(NaN, length(b))))
    B = Functional((u, θ) -> ∫(u)dΓ, Sbad; dJdu = u -> (w -> ∫(w)dΓ), scale = 1.0)
    dir2 = mktempdir()
    @test_throws OptimizationFailure solve!(Optimization(z; objective = B, constraints = (volume = q.Vol,)),
                                            DensityMethod(OC(); metric = Euclidean()); maxiter = 3,
                                            verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 2), checkpoint = Checkpoint(filepath = dir2, every = 10), stop = nothing)
    @test isfile(joinpath(dir2, "checkpoint.toml"))
  end

  @testset "Phases seen by the user's solve (SiMPL, non-uniform start, jump)" begin
    phases = Tuple{Int,Int,Symbol}[]
    q = compliance_problem(; init = nothing, fraction = 0.7, phases)
    r = solve!(q.problem, DensityMethod(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 5,
               continuation = Continuation(β = q.H => [2.0, 4.0]; every = 3), verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
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
    run(q, maxiter; kwargs...) = solve!(q.problem, DensityMethod(OC(); metric = Euclidean()); maxiter, continuation = sched(q),
                                        verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), kwargs..., stop = nothing)
    qa = compliance_problem(); ra = run(qa, 9)
    qb = compliance_problem(); run(qb, 5; checkpoint = Checkpoint(filepath = dir, every = 5))
    ck = load_checkpoint(dir)
    @test ck.stage == 2 && ck.stage_iterations == 2
    qc = compliance_problem(); rc = run(qc, 9; resume = ck)
    @test get_values(qc.z) == get_values(qa.z)
    @test rc.history.continuation == ra.history.continuation
    qd = compliance_problem()
    @test_throws ArgumentError solve!(qd.problem, DensityMethod(OC(); metric = Euclidean()); maxiter = 9, resume = ck,
                                      continuation = Continuation(β = qd.H => [1.0]; every = 3), verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
  end

  @testset "Objective criterion counts only the current stage" begin
    q = compliance_problem()
    r = solve!(q.problem, DensityMethod(OC(); metric = Euclidean()); maxiter = 40, stop = StopCriteria(objective_range = 5e-2, window = 2),
               continuation = Continuation(β = q.H => [1.0, 2.0, 4.0, 8.0]; every = 20, close_early = true), verbose = 0,
               on_failure = RejectStep(shrink = 0.5, max_attempts = 3))
    c = r.history.continuation.β
    lengths = [count(==(v), c) for v in unique(c)]
    @test all(>=(2), lengths[1:end-1])          # window iterations at least before a stage can end
    @test StopCriteria(objective_range = 1e-3).window == 10
    @test_throws ArgumentError StopCriteria(objective_range = 1e-3, window = 1)
  end

  @testset "SiMPL without constraint, loop bookkeeping without allocations" begin
    q = compliance_problem()
    r = solve!(Optimization(q.z; objective = q.C), DensityMethod(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 4,
               verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
    @test r.history.objective[end] < r.history.objective[1] && isempty(r.history.constraints)
    @test isempty(r.history.multipliers) && r.start_scaling === nothing
    h = OptHistory(10, (:β,); constraints = (:volume,), monitors = (:grey,))
    buf = [1.0]
    row = (1, 1.0, NaN, NaN, 0.1, 0.2, 1, 1, 0, 2)
    _record!(h, row, (0.0,), (0.5,), (NaN,), buf)
    @test (@allocated _record!(h, row, (0.0,), (0.5,), (NaN,), buf)) == 0
    st = StopCriteria(change_tol = 1e-3, objective_range = 1e-3, window = 2)
    info = HyperShape.Optimizers.StepInfo(true, 1.0, 1.0, NaN, 0.1, 0.1, 0.0, 1, 0.0, NaN)
    _stop_reason(st, info, h, 0)
    @test (@allocated _stop_reason(st, info, h, 0)) == 0
    @test_throws ArgumentError Continuation(step = q.H => [1.0]; every = 2)    # history column name
    @test_throws ArgumentError Continuation(β = q.H => [1.0, -2.0]; every = 2)
  end

  @testset "More review cases" begin
    # a projected volume from a uniform 0.5 start with η = 0.5 on a finer mesh is not linear
    q = compliance_problem(; n = 40, init = 0.5)
    set_beta!(q.H, 3.0)
    Vp = Functional(θ -> sum(∫(θ.ρ)q.dΩ) - 0.4, q.θ; dJdθ = θ -> (δ -> ∫(δ)q.dΩ), scale = 0.4, linear = true)
    o = FunctionalOracle(Optimization(q.z; objective = q.C, constraints = (volume = Vp,)), Euclidean())
    @test_throws ArgumentError _setup_constraint!(o, 1, q.problem.ctx)
    # a start with values at the bounds: the first solve of SiMPL is the current design
    phases = Tuple{Int,Int,Symbol}[]
    q = compliance_problem(; init = [isodd(i) ? 0.0 : 0.8 for i in 1:64], phases)
    solve!(q.problem, DensityMethod(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 1, verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
    @test phases[findfirst(p -> p[1] == 1, phases)][3] === :optimization
    # a resumed run stops where the uninterrupted one does
    dir = mktempdir()
    stopc = StopCriteria(objective_range = 2e-2, window = 2)
    runs(q, maxiter; kwargs...) = solve!(q.problem, DensityMethod(OC(); metric = Euclidean()); maxiter, stop = stopc,
                                         verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), kwargs...)
    ra = runs(compliance_problem(), 40)
    @test ra.status === :converged
    kstop = ra.iterations
    runs(compliance_problem(), kstop - 1; checkpoint = Checkpoint(filepath = dir, every = 100))
    rc = runs(compliance_problem(), 40; resume = load_checkpoint(dir))
    @test rc.iterations == kstop
    # a jump whose first design fails is undone: β and the checkpoint stay in the previous stage
    hold = Ref{Any}(nothing)
    qf2 = compliance_problem(; solve_flag = () -> hold[].β != 2.0)
    hold[] = qf2.H
    dir2 = mktempdir()
    @test_throws OptimizationFailure solve!(qf2.problem, DensityMethod(OC(); metric = Euclidean()); maxiter = 6,
                                            continuation = Continuation(β = qf2.H => [1.0, 2.0]; every = 2),
                                            checkpoint = Checkpoint(filepath = dir2, every = 100), verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 2), stop = nothing)
    ck = load_checkpoint(dir2)
    @test ck.stage == 1 && ck.iteration == 2 && qf2.H.β == 1.0
    # a fixed scale is kept on resume
    q = compliance_problem()
    rq = solve!(q.problem, DensityMethod(OC(); metric = Euclidean()); maxiter = 2, checkpoint = Checkpoint(filepath = dir, every = 100),
                verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
    q2 = compliance_problem()
    q2.Vol.scale[] = 7.0
    solve!(q2.problem, DensityMethod(OC(); metric = Euclidean()); maxiter = 3, resume = load_checkpoint(dir),
           verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
    @test q2.Vol.scale[] == 7.0
    # a user's file in the checkpoint folder is not deleted
    write(joinpath(dir, "z_notes.txt"), "keep")
    solve!(compliance_problem().problem, DensityMethod(OC(); metric = Euclidean()); maxiter = 2, checkpoint = Checkpoint(filepath = dir, every = 1),
           verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
    @test isfile(joinpath(dir, "z_notes.txt"))
  end

  @testset "Decisions of Jesús: stalls end a stage, infeasible starts, multipliers" begin
    # no step at β = 1 (wrong gradient from iteration 3): the stage ends, β goes to 2, and the run stalls there
    sign = Ref(1.0)
    q = compliance_problem(; sign)
    r = solve!(q.problem, DensityMethod(SiMPL(max_backtracks = 1); metric = LumpedMass(q.dΩ)); maxiter = 10,
               continuation = Continuation(β = q.H => [1.0, 2.0]; every = 5), verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3),
               post = OptPostProcessor((_, ctx) -> (ctx.iteration == 2 && (sign[] = -1.0)); every = 1, initial = false), stop = nothing)
    @test r.status === :stalled && r.iterations == 2 && q.H.β == 2.0
    # an infeasible start is scaled towards the lower bound and SiMPL runs
    q = compliance_problem(; init = 0.6)
    io = IOBuffer()
    r = @test_logs solve!(q.problem, DensityMethod(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 5, verbose = 0, io,
                          on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)        # level 0: no logs, nothing on io
    @test isempty(take!(io)) && r.start_scaling ≈ 0.4 / 0.6
    @test r.iterations == 5 && all(<=(1e-8), r.history.constraints.volume)
    @test nonincreasing(r.history.objective)
    # multipliers in the history
    @test all(>=(0), r.history.multipliers.volume)
    q = compliance_problem()
    r = solve!(q.problem, DensityMethod(OC(); metric = Euclidean()); maxiter = 3, verbose = 0, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
    @test all(>(0), r.history.multipliers.volume)
  end

  @testset "Second review: stalls, solve counts, old checkpoints, scaled start" begin
    # a stall ends a stage and the run goes on; the summary counts every forward solve of the call
    q = compliance_problem(; n = 16)
    c0 = q.nforward[]
    io = IOBuffer()
    r = solve!(q.problem, DensityMethod(SiMPL(max_backtracks = 0); metric = LumpedMass(q.dΩ)); maxiter = 30, io,
               continuation = Continuation(β = q.H => [1.0, 2.0, 4.0, 8.0]; every = 6), on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
    out = String(take!(io))
    @test occursin("the stage ends", out)
    ends = findfirst("the stage ends", out)
    @test count(l -> occursin(r"^\s+\d+\s+\d\.\d{6}e[+-]\d\d", l), split(out[last(ends):end], '\n')) >= 1   # rows after it
    m = match(r"solves: forward (\d+)", out)
    @test parse(Int, m[1]) == q.nforward[] - c0
    @test !occursin(r"─\n\s+it .*\n\s+─+\n\s+─", out)                   # no header without rows
    # checkpoint: history by name, resume from the folder, checkpoints of other versions refused
    dir = mktempdir()
    q = compliance_problem()
    rb = solve!(q.problem, DensityMethod(OC(); metric = Euclidean()); maxiter = 2, verbose = 0,
                checkpoint = Checkpoint(filepath = dir, every = 2), stop = nothing)
    ck = load_checkpoint(dir)
    @test ck.history isa OptHistory && ck.history.multipliers.volume == rb.history.multipliers.volume
    @test ck.history.constraints.volume == rb.history.constraints.volume
    @test ck.history.stationarity == rb.history.stationarity && all(isfinite, ck.history.stationarity)
    @test keys(ck.scales) == (:objective, :volume)
    r1 = solve!(compliance_problem().problem, DensityMethod(OC(); metric = Euclidean()); maxiter = 4, verbose = 0,
                resume = dir, stop = nothing)
    @test r1.iterations == 4 && r1.history.multipliers.volume[1:2] == rb.history.multipliers.volume
    tfile = joinpath(dir, "checkpoint.toml")
    write(tfile, replace(read(tfile, String), "hypershape_checkpoint = 3" => "format = 5"))
    @test_throws ArgumentError load_checkpoint(dir)
    @test_throws ArgumentError load_checkpoint(mktempdir())
    # the scaled start leaves the entries the constraint does not act on
    q = compliance_problem()
    nz = length(get_values(q.z))
    a = ones(nz); a[1:5] .= 0
    lb = zeros(nz)
    zk = fill(0.6, nz)
    of = FunctionalOracle(q.problem, Euclidean())
    of.a[1] = a; of.V[1] = 0.4 * sum(a); of.lb .= lb
    t = HyperShape.Methods._make_feasible!(of, zk)
    @test t ≈ 0.4 / 0.6 && all(==(0.6), zk[1:5]) && dot(a, zk) ≈ 0.4 * sum(a)
    # printing at level 0 allocates nothing
    h = OptHistory(4, (:β,); constraints = (:volume,), monitors = (:grey,))
    _record!(h, (1, 1.0, NaN, NaN, 0.1, 0.2, 1, 1, 0, 2), (0.0,), (0.5,), (NaN,), [1.0])
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
    method = name === :OC ? DensityMethod(OC()) : DensityMethod(SiMPL(); metric = LumpedMass(q.dΩ))
    io = IOBuffer()
    r = solve!(Optimization(q.z; objective = q.C, constraints = (volume = V,)), method; maxiter = 30, io, verbose = 1,
               continuation = Continuation(β = q.H => [3.0, 8.0]; every = 15), stop = nothing)
    out = String(take!(io))
    gs = r.history.constraints.volume
    @test occursin("constraint: volume, nonlinear (tol $(name === :OC ? "1e-04" : "1e-06"))", out)
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
    run(q, maxiter; kwargs...) = solve!(Optimization(q.z; objective = q.C, constraints = (volume = Volume(q.θ, q.dΩ; max_fraction = 0.4),)),
                                        DensityMethod(OC()); maxiter, verbose = 0, stop = nothing, kwargs...)
    qa = compliance_problem()
    ra = run(qa, 8)
    qb = compliance_problem()
    run(qb, 4; checkpoint = Checkpoint(filepath = dir, every = 2))
    qc = compliance_problem()
    rc = run(qc, 8; resume = load_checkpoint(dir))
    @test rc.history.iteration == 1:8 && rc.history.objective[1:4] == ra.history.objective[1:4]
    @test abs(rc.history.constraints.volume[end]) <= 1e-4
    # move limits and correction shift are saved: the resumed run is the uninterrupted one
    @test rc.history.objective == ra.history.objective
  end

  @testset "Volume of the projected density: Taylor test (P1)" begin
    q = compliance_problem(; design = :P1)
    @test check_consistency(Volume(q.θ, q.dΩ; max_fraction = 0.4)).passed
  end

  @testset "Volume: infeasible start, linear volume of the design, errors" begin
    q = compliance_problem(; init = 0.9)
    vol = sum(∫(1.0)q.dΩ)
    V = Volume(q.θ, q.dΩ; max_fraction = 0.4)
    r = solve!(Optimization(q.z; objective = q.C, constraints = (volume = V,)), DensityMethod(SiMPL(); metric = LumpedMass(q.dΩ));
               maxiter = 3, verbose = 0, stop = nothing)
    @test r.start_scaling !== nothing && 0 < r.start_scaling < 1
    @test all(<=(0), r.history.constraints.volume)
    # the volume of the design is linear: the same run as the hand-written linear functional
    runs = map((:volume, :hand)) do kind
      q = compliance_problem()
      C = kind === :volume ? Volume(q.x, q.dΩ; max_fraction = 0.4) : q.Vol
      solve!(Optimization(q.z; objective = q.C, constraints = (volume = C,)), DensityMethod(OC()); maxiter = 6, verbose = 0, stop = nothing)
    end
    @test runs[1].history.objective ≈ runs[2].history.objective rtol = 1e-12
    @test_throws ArgumentError Volume(q.θ, q.dΩ; max_fraction = -1.0)
    @test_throws ArgumentError Volume(q.θ, q.dΩ; max_fraction = 1.5)
    @test_throws ArgumentError Volume(q.θ, q.dΩ; max_fraction = 0.5, field = :E)
    @test_throws ArgumentError Optimization(q.z; objective = q.C, constraints = (a = V, b = V))
  end

  @testset "Stopping: range of the objective, kkt_rtol, the grey monitor" begin
    h = OptHistory(30, ())
    rec!(f; kkt = NaN) = _record!(h, (h.n + 1, f, kkt, NaN, 0.1, 0.1, 1, 1, 0, 1), (), (), (), ())
    info(kkt = NaN) = HyperShape.Optimizers.StepInfo(true, 1.0, 1.0, kkt, 0.1, 0.1, 0.0, 1, NaN, NaN)
    st = StopCriteria(objective_range = 1e-3, window = 4)
    for f in (1.0, 1.01, 1.0, 1.01, 1.0)          # an oscillation: the ends of the window agree, the range does not
      rec!(f)
    end
    @test _stop_reason(st, info(), h, 0) === nothing
    @test _stop_reason(st, info(), h, 2) === nothing   # only 3 iterations of the stage
    for f in (0.9, 0.9004, 0.9002, 0.9001)
      rec!(f)
    end
    @test _stop_reason(st, info(), h, 0) === :objective_range
    # an objective tending to zero: floor 1e-3 |f₁|; a negative objective
    h0 = OptHistory(10, ())
    for f in (1.0, 1e-6, 0.8e-6, 0.6e-6, 0.5e-6)
      _record!(h0, (h0.n + 1, f, NaN, NaN, 0.1, 0.1, 1, 1, 0, 1), (), (), (), ())
    end
    @test _stop_reason(StopCriteria(objective_range = 1e-3, window = 4), info(), h0, 0) === :objective_range
    hn = OptHistory(10, ())
    for f in (-1.0, -2.0, -2.0005, -2.0004)
      _record!(hn, (hn.n + 1, f, NaN, NaN, 0.1, 0.1, 1, 1, 0, 1), (), (), (), ())
    end
    @test _stop_reason(StopCriteria(objective_range = 1e-3, window = 3), info(), hn, 0) === :objective_range
    @test _stop_reason(StopCriteria(objective_range = 1e-3, window = 4), info(), hn, 0) === nothing
    # kkt relative to the first iteration
    hk = OptHistory(10, ())
    # stationarity_rtol: reference = first finite value (NaN rows are skipped)
    hs = OptHistory(10, ())
    for (k, sv) in enumerate((NaN, NaN, 1.0, 0.5, 0.05))
      _record!(hs, (k, 1.0, NaN, sv, 0.1, 0.1, 1, 1, 0, 1), (), (), (), ())
    end
    sinfo(S) = HyperShape.Optimizers.StepInfo(true, 1.0, 1.0, NaN, 0.1, 0.1, 0.0, 1, NaN, S)
    @test _stop_reason(StopCriteria(stationarity_rtol = 0.1), sinfo(0.05), hs, 0) === :stationarity_rtol
    @test _stop_reason(StopCriteria(stationarity_rtol = 0.01), sinfo(0.05), hs, 0) === nothing
    @test _stop_reason(StopCriteria(stationarity_tol = 0.1), sinfo(0.05), hs, 0) === :stationarity_tol
    @test _stop_reason(StopCriteria(stationarity_tol = 0.1), sinfo(0.05), hs, hs.n - 1) === nothing   # first of a stage
    hn2 = OptHistory(10, ())
    _record!(hn2, (1, 1.0, NaN, NaN, 0.1, 0.1, 1, 1, 0, 1), (), (), (), ())
    _record!(hn2, (2, 1.0, NaN, NaN, 0.1, 0.1, 1, 1, 0, 1), (), (), (), ())
    @test _stop_reason(StopCriteria(stationarity_rtol = 0.1), sinfo(NaN), hn2, 0) === nothing
    _record!(hk, (1, 1.0, 10.0, NaN, 0.1, 0.1, 1, 1, 0, 1), (), (), (), ())
    _record!(hk, (2, 1.0, 0.05, NaN, 0.1, 0.1, 1, 1, 0, 1), (), (), (), ())
    @test _stop_reason(StopCriteria(kkt_rtol = 1e-2), info(0.05), hk, 0) === :kkt_rtol
    @test _stop_reason(StopCriteria(kkt_rtol = 1e-2), info(0.05), hk, 1) === nothing   # first iteration of a stage
    @test _stop_reason(StopCriteria(kkt_rtol = 1e-3), info(0.05), hk, 0) === nothing
    # the default criterion is reached on the small problem
    for (mk, lumped) in ((() -> SiMPL(), true), (() -> OC(), false))
      q = compliance_problem()
      method = lumped ? DensityMethod(mk(); metric = LumpedMass(q.dΩ)) : DensityMethod(mk())
      r = solve!(q.problem, method; maxiter = 300, verbose = 0)
      @test r.status === :converged && r.reason === :objective_range && r.iterations < 300
    end
    # grey monitor: history, table and closing line
    q = compliance_problem()
    io = IOBuffer()
    r = solve!(q.problem, DensityMethod(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 4, stop = nothing, io,
               monitors = (grey = Grey(q.θ, q.dΩ),))
    out = String(take!(io))
    ρ = q.θ.ρ
    @test r.history.monitors.grey[end] ≈ 4 * sum(∫(ρ * (1 - ρ))q.dΩ) / sum(∫(1.0)q.dΩ)
    @test all(g -> 0 <= g <= 1, r.history.monitors.grey)
    @test occursin("grey", out) && occursin(", grey ", out)
    @test (solve!(compliance_problem().problem, DensityMethod(OC()); maxiter = 2, verbose = 0, stop = nothing).history.monitors |> isempty)
    @test_throws ArgumentError solve!(q.problem, DensityMethod(OC()); maxiter = 2, verbose = 0, monitors = (grey = q.θ,))
    θ2 = PhysicalFields(q.x; ρ = Identity(), E = :ρ => Identity())
    @test_throws ArgumentError Grey(θ2, q.dΩ)
    @test Grey(θ2, q.dΩ; field = :E).field === :E
    @test_throws MethodError Grey(q.x, q.dΩ)
    @test_throws ArgumentError Continuation(kkt = q.H => [1.0, 2.0]; every = 2)   # a history column
    # grey through a checkpoint, P1 design
    dir = mktempdir()
    q1 = compliance_problem(; design = :P1)
    r1 = solve!(q1.problem, DensityMethod(OC()); maxiter = 3, verbose = 0, stop = nothing, monitors = (grey = Grey(q1.θ, q1.dΩ),),
                checkpoint = Checkpoint(filepath = dir, every = 3))
    @test all(isfinite, r1.history.monitors.grey) && load_checkpoint(dir).history.monitors.grey == r1.history.monitors.grey
    # banner: no "last stage only" without criteria
    io = IOBuffer()
    solve!(compliance_problem().problem, DensityMethod(OC()); maxiter = 1, io, verbose = 1, stop = StopCriteria(),
           continuation = Continuation(β = q.H => [1.0, 2.0]; every = 2))
    @test !occursin("last stage only", String(take!(io)))
  end

  @testset "Defaults: metric of OC, stopping criteria, failure handling and the banner" begin
    default_stop = HyperShape.Methods.default_stop
    @test DensityMethod(OC()).default_metric && !DensityMethod(OC(); metric = Euclidean()).default_metric
    @test RejectStep() === RejectStep(shrink = 0.5, max_attempts = 5)
    for o in (OC(), SiMPL())
      d = default_stop(o)
      @test d.objective_range == 1e-3 && d.window == 10 && d.kkt_tol === nothing && d.kkt_rtol === nothing && d.change_tol === nothing
    end
    # the defaults give the same run as the same values written explicitly
    for (mkopt, lumped) in ((() -> OC(), false), (() -> SiMPL(), true))
      runs = map((true, false)) do dflt
        q = compliance_problem()
        method = lumped ? DensityMethod(mkopt(); metric = LumpedMass(q.dΩ)) : DensityMethod(mkopt())
        io = IOBuffer()
        r = dflt ? solve!(q.problem, method; maxiter = 8, io, verbose = 1) :
                   solve!(q.problem, method; maxiter = 8, io, verbose = 1, on_failure = RejectStep(shrink = 0.5, max_attempts = 5),
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
    solve!(compliance_problem().problem, DensityMethod(OC()); maxiter = 2, io, verbose = 1, stop = nothing)
    out = String(take!(io))
    @test occursin("metric: Euclidean (default) (OC: only for the stationarity measure)", out) && occursin("stop: maxiter only · maxiter 2", out)
  end

  @testset "Extension points: Parameter, metric of the user, stop function" begin
    # a Parameter as a dependency of a state: solved again only when it changes
    q = compliance_problem()
    prm = Parameter(1.0)
    S2 = StateOperator(StaticLinearModel(q.l, q.a, q.U, q.V, q.bc; ls = LUSolver()), (q.θ, prm))
    HyperShape.Operators.apply!(S2, nothing)
    n0 = HyperShape.Operators.num_solves(S2)
    HyperShape.Operators.apply!(S2, nothing)
    @test HyperShape.Operators.num_solves(S2) == n0
    prm[] = 2.0
    HyperShape.Operators.apply!(S2, nothing)
    @test HyperShape.Operators.num_solves(S2) == n0 + 1
    # continued by Continuation
    prm[] = 1.0
    r = solve!(q.problem, DensityMethod(OC()); maxiter = 6, verbose = 0, stop = nothing,
               continuation = Continuation(p = prm => [1.0, 2.0, 3.0]; every = 2))
    @test r.history.continuation.p == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0] && prm[] == 3.0
    @test_throws ArgumentError Continuation(p = prm => [1.0, NaN]; every = 2)
    # metric: a vector of weights or an object with metric_weights gives the same run as LumpedMass
    qm0 = compliance_problem()
    m = HyperShape.Methods.FunctionalOracle(qm0.problem, LumpedMass(qm0.dΩ)).metric
    runs = map((qq -> LumpedMass(qq.dΩ), qq -> copy(m), qq -> _UserMetric(m))) do mk
      qq = compliance_problem()
      solve!(qq.problem, DensityMethod(SiMPL(); metric = mk(qq)); maxiter = 4, verbose = 0, stop = nothing).history.objective
    end
    @test runs[1] == runs[2] == runs[3]
    @test_throws ArgumentError DensityMethod(SiMPL(); metric = :lumped)
    @test_throws ArgumentError solve!(compliance_problem().problem, DensityMethod(SiMPL(); metric = ones(3)); maxiter = 2, verbose = 0)
    @test_throws ArgumentError solve!(compliance_problem().problem, DensityMethod(SiMPL(); metric = -copy(m)); maxiter = 2, verbose = 0)
    io = IOBuffer()
    solve!(compliance_problem().problem, DensityMethod(SiMPL(); metric = _UserMetric(m)); maxiter = 1, verbose = 1, io, stop = nothing)
    @test occursin("metric: _UserMetric", String(take!(io)))
    # stop as a function of the user
    io = IOBuffer()
    r = solve!(compliance_problem().problem, DensityMethod(OC()); maxiter = 10, io, verbose = 1,
               stop = (info, h, ctx) -> h.n >= 3 && ctx.iteration == 3)
    out = String(take!(io))
    @test r.status === :converged && r.reason === :custom && r.iterations == 3
    @test occursin("stop: function of the user", out) && occursin("criterion of the user met", out)
    @test_throws ArgumentError solve!(compliance_problem().problem, DensityMethod(OC()); maxiter = 2, verbose = 0, stop = 1e-3)
    # the history the function receives is trimmed to the iterations done; a non-Bool result is an error
    seen = Int[]
    r = solve!(compliance_problem().problem, DensityMethod(OC()); maxiter = 6, verbose = 0,
               stop = (info, h, ctx) -> (push!(seen, length(h.objective) - h.n); all(isfinite, h.objective) && h.n >= 2))
    @test r.reason === :custom && r.iterations == 2 && all(==(0), seen)
    @test_throws ArgumentError solve!(compliance_problem().problem, DensityMethod(OC()); maxiter = 3, verbose = 0,
                                      stop = (info, h, ctx) -> 1)
    # a callable object
    r = solve!(compliance_problem().problem, DensityMethod(OC()); maxiter = 6, verbose = 0, stop = _StopAt(3))
    @test r.reason === :custom && r.iterations == 3
    # metric_weights with a typed z
    qt = compliance_problem()
    rt = solve!(qt.problem, DensityMethod(SiMPL(); metric = _TypedMetric(copy(m))); maxiter = 2, verbose = 0, stop = nothing)
    @test rt.iterations == 2
    @test_throws ArgumentError solve!(compliance_problem().problem, DensityMethod(SiMPL(); metric = _StopAt(1)); maxiter = 2, verbose = 0)
    # the same object continued under two names
    @test_throws ArgumentError Continuation(a = prm => [1.0, 2.0], b = prm => [3.0, 4.0]; every = 2)
    # a Parameter read by the functional only: declared in its parameters, the value follows it
    qw = compliance_problem()
    w = Parameter(1.0)
    dΩw = qw.dΩ
    Fw = Functional(θ -> w[] * sum(∫(θ.ρ)dΩw), qw.θ; dJdθ = θ -> (δ -> ∫(w[] * δ)dΩw), scale = 1.0, parameters = (w,))
    Fn = Functional(θ -> w[] * sum(∫(θ.ρ)dΩw), qw.θ; dJdθ = θ -> (δ -> ∫(w[] * δ)dΩw), scale = 1.0)
    ctxw = OptContext(:optimization)
    v1, n1 = functional_value(Fw, ctxw), functional_value(Fn, ctxw)
    w[] = 2.0
    @test functional_value(Fw, ctxw) ≈ 2 * v1 && functional_value(Fn, ctxw) == n1   # without parameters: stale
    @test_throws ArgumentError Functional(θ -> 0.0, qw.θ; dJdθ = θ -> (δ -> ∫(δ)dΩw), scale = 1.0, parameters = (1.0,))
  end

  @testset "Screen output" begin
    io = IOBuffer()
    q = compliance_problem(; init = 0.6)
    cnt = Ref(0)
    r = solve!(q.problem, DensityMethod(SiMPL(); metric = LumpedMass(q.dΩ)); maxiter = 6, io,
               stop = StopCriteria(kkt_tol = 1e-12), continuation = Continuation(β = q.H => [1.0, 2.0]; every = 3),
               on_failure = RejectStep(shrink = 0.5, max_attempts = 3))
    out = String(take!(io))
    @test occursin("HyperShape · DensityMethod(SiMPL)", out) && occursin("continuation β: 1 → 2", out)
    @test occursin("scaled towards the lower bound", out)
    @test occursin("objective", out) && occursin("f/a/F", out)
    @test count(l -> occursin(r"^\s+\d+\s+\d\.\d{6}e[+-]\d\d", l), split(out, '\n')) == 6   # one row per iteration
    @test occursin("→ β = 2 (stage 2 of 2)", out)
    @test occursin("⚠ maximum number of iterations (6)", out) && occursin("solves: forward", out)
    # level 1: no rows
    io = IOBuffer()
    q = compliance_problem()
    solve!(q.problem, DensityMethod(OC(); metric = Euclidean()); maxiter = 3, verbose = 1, io, stop = StopCriteria(change_tol = 1.0),
           on_failure = RejectStep(shrink = 0.5, max_attempts = 3))
    out = String(take!(io))
    @test occursin("✓ converged", out) && !occursin("f/a/F", out)
    # a rejected step and a failure
    io = IOBuffer()
    cnt[] = 0
    q = compliance_problem(; solve_flag = () -> (cnt[] += 1; cnt[] != 3 && cnt[] < 6))
    @test_throws OptimizationFailure solve!(q.problem, DensityMethod(OC(); metric = Euclidean()); maxiter = 6, io,
                                            on_failure = RejectStep(shrink = 0.5, max_attempts = 2), stop = nothing)
    out = String(take!(io))
    @test occursin("✗ iteration 2: the forward problem did not converge", out) && occursin("(2 attempts)", out)
    @test occursin("✗ failed after", out)
    @test_throws ArgumentError solve!(compliance_problem().problem, DensityMethod(OC(); metric = Euclidean()); maxiter = 2,
                                      verbose = 5, on_failure = RejectStep(shrink = 0.5, max_attempts = 3), stop = nothing)
  end

  @testset "Resume: changed bound, memory of the step, state after an exception, SiMPL with AL" begin
    # a linear constraint whose bound is changed on resume follows the new bound (the saved offset is not used)
    mk(frac) = begin
      q = compliance_problem()
      dΩ = q.dΩ
      q = (; q..., vol = sum(∫(1.0)dΩ))
      Vraw = Functional(θ -> sum(∫(θ.ρ)dΩ), q.θv; dJdθ = θ -> (δ -> ∫(δ)dΩ), scale = 1.0, linear = true)
      (q, Optimization(q.z; objective = q.C, constraints = (volume = Constraint(Vraw; max = frac * q.vol),)), Vraw)
    end
    dir = mktempdir()
    q, pb, _ = mk(0.4)
    solve!(pb, DensityMethod(OC()); maxiter = 4, verbose = 0, stop = nothing, checkpoint = Checkpoint(filepath = dir, every = 4))
    q, pb, Vraw = mk(0.3)
    solve!(pb, DensityMethod(OC()); maxiter = 30, verbose = 0, stop = nothing, resume = dir)
    @test functional_value(Vraw, nothing) / q.vol <= 0.3 + 1e-3

    # the memory of the step: copied in its order, numbers as Float64; wrong entries fail when captured
    cm = HyperShape.Methods._capture_memory
    v = Float32[1, 2, 3, 4]
    m = cm((v = v, k = 3, flag = true), 4, OC())
    @test keys(m) == (:v, :k, :flag) && m.v isa Vector{Float64} && m.v == v && m.k === 3.0 && m.flag === 1.0
    v[1] = 9
    @test m.v[1] == 1.0
    @test_throws ArgumentError cm((w = [1.0, 2.0],), 4, OC())
    @test_throws ArgumentError cm((t = (1.0, 2.0),), 4, OC())
    @test_throws ArgumentError cm((n = nothing,), 4, OC())
    @test_throws ArgumentError cm([1.0], 4, OC())
    st = OptState(1, 0, 1, collect(1.0:4), [:ρ], Dict(:ρ => 0), Symbol[], (objective = 1.0,), OptHistory(2), "M", "X",
                  (v = [1.0, 2, 3, 4], k = 3.0, w = [4.0, 3, 2, 1]), Dict{Symbol,Symbol}(), nothing, Dict{Symbol,Float64}(),
                  "FixedMesh", 0)
    d2 = mktempdir()
    HyperShape.Methods._write_checkpoint(Checkpoint(filepath = d2, every = 1), st)
    @test isequal(load_checkpoint(d2).memory, st.memory)
    write(joinpath(d2, "memory_1.bin"), zeros(3))
    @test_throws ArgumentError load_checkpoint(d2)

    # SiMPL with an augmented Lagrangian (μ growing): resumed from a file, twice, and from problem.state[]
    mkal() = begin
      p = state_problem(; n = 16, init = fill(0.4, 256))
      B0 = functional_value(p.boundary(scale = 1.0), nothing)
      pb = Optimization(objective = p.compliance(scale = :initial),
                        constraints = (volume = Volume(p.θ, p.dΩ; max_fraction = 0.4),
                                       boundary = Constraint(p.boundary(scale = :bound); min = 0.5B0)))
      p, pb
    end
    lag = AugmentedLagrangian(Bertsekas(μ0 = 0.5, growth = 1.5, window = 3, progress = Inf, μmax = 50.0))
    run(p, pb, maxiter; kw...) = solve!(pb, DensityMethod(SiMPL(); metric = LumpedMass(), lagrangian = lag); maxiter,
                                        verbose = 0, stop = nothing, kw...)
    p, pb = mkal(); ra = run(p, pb, 12); za = copy(get_values(pb.z))
    d3 = mktempdir()
    p, pb = mkal(); run(p, pb, 4; checkpoint = Checkpoint(filepath = d3, every = 4)); st4 = pb.state[]
    # the state in memory and the one read from the file are the same, field by field
    ck = load_checkpoint(d3)
    for f in fieldnames(OptState)
      f === :history || @test isequal(getfield(ck, f), getfield(st4, f))
    end
    @test isequal(ck.history.objective, st4.history.objective) && isequal(ck.history.penalty_values, st4.history.penalty_values)
    p, pb = mkal(); rb = run(p, pb, 12; resume = d3)
    @test get_values(pb.z) == za && rb.history.objective == ra.history.objective
    @test isequal(rb.history.multipliers.boundary, ra.history.multipliers.boundary) && rb.multipliers == ra.multipliers
    p, pb = mkal(); run(p, pb, 8; resume = st4, checkpoint = Checkpoint(filepath = d3, every = 4))
    p, pb = mkal(); run(p, pb, 12; resume = d3)
    @test get_values(pb.z) == za

    # after an exception: the state of the last accepted iteration, without the memory of the step, and no
    # message about a checkpoint when there is none; it can be resumed
    q = compliance_problem()
    buf = IOBuffer()
    @test_throws ErrorException solve!(q.problem, DensityMethod(SiMPL(); metric = LumpedMass()); maxiter = 8, verbose = 1,
                                       io = buf, stop = (info, h, ctx) -> h.n >= 4 ? error("boom") : false)
    st = q.problem.state[]
    @test st isa OptState && st.memory === nothing && st.iteration == length(st.history) && st.iteration in (3, 4)
    @test !occursin("checkpoint", String(take!(buf)))
    r = solve!(q.problem, DensityMethod(SiMPL(); metric = LumpedMass()); maxiter = 8, verbose = 0, stop = nothing, resume = st)
    @test length(r.history) == 8
  end

  @testset "Factories and rebuilds: Optimization(model; design_setup, physics_setup), RebuildCheck" begin
    RC = HyperShape.Methods.RebuildCheck
    model = mbb_model(24, 8)
    method(opt) = DensityMethod(opt === :OC ? OC() : SiMPL(); metric = LumpedMass())
    # a run of 24 iterations, β = 1, 2, 4 every 8 (a jump after iterations 8 and 16)
    function run(; opt = :SiMPL, order = 0, linear = false, al = false, frozen = false, maxiter = 24, verbose = 0, kw...)
      proj = TanhProjection(β = 1.0, η = 0.5)
      ds, ps = mbb_factories(; order, linear, al, frozen, proj, output = false)
      pb = Optimization(model; design_setup = ds, physics_setup = ps)
      r = solve!(pb, method(opt); maxiter, verbose, stop = nothing,
                 continuation = linear ? nothing : Continuation(β = proj => [1.0, 2.0, 4.0]; every = 8), kw...)
      pb, r
    end

    # the problem built by the factories: names kept, the same run as the problem built from its functionals
    pb, r = run(; maxiter = 6)
    @test r.problem === pb && pb.setup !== nothing
    @test haskey(pb.physics, :θ) && pb.physics.θ isa PhysicalFields && pb.design.x isa DesignFields
    proj = TanhProjection(β = 1.0, η = 0.5)
    ds, ps = mbb_factories(; proj, output = false)
    d = ds(model)
    p = ps(model, d.x)
    plain = Optimization(d.z; objective = p.objective, constraints = p.constraints)
    rp = solve!(plain, method(:SiMPL); maxiter = 6, verbose = 0, stop = nothing, monitors = p.monitors,
                continuation = Continuation(β = proj => [1.0, 2.0, 4.0]; every = 8))
    @test same_history(rp.history, r.history) && get_values(rp.z) == get_values(r.z)
    @test rp.problem === plain && plain.setup === nothing && plain.physics.objective === p.objective
    rep = check_consistency(pb)
    @test rep.passed && length(rep.results) >= 2

    # RebuildCheck gives the run of FixedMesh bit for bit (solve counts aside), before a jump and in the middle of a
    # stage, both factories or the physics only, OC and SiMPL, P0 and P1, linear and projected volume, augmented
    # Lagrangian, frozen zone, several events
    cases = [(; opt = :SiMPL), (; opt = :OC), (; opt = :SiMPL, order = 1), (; opt = :OC, order = 1),
             (; opt = :SiMPL, linear = true), (; opt = :SiMPL, al = true), (; opt = :SiMPL, frozen = true)]
    for c in cases
      _, ref = run(; c...)
      for d in (RC(at = 8), RC(at = 11), RC(at = 11, what = :physics), RC(every = 5))
        pbe, re = run(; c..., discretization = d)
        ok = same_history(re.history, ref.history) && get_values(re.z) == get_values(ref.z)
        ok || @info "RebuildCheck differs" c d
        @test ok
        @test re.problem !== pbe                           # the last problem, built by the last rebuild
        @test pbe.state[].events == (d.every > 0 ? 4 : 1)
        @test (d.what === :physics) == (re.problem.z === pbe.z)
        @test sum(re.history.forward) == sum(ref.history.forward)     # the solves of a rebuild are not in the rows
        @test pbe.state[] === re.problem.state[]
      end
    end

    # banner, event line and summary; a factory that captures something that changes gives "differs"
    buf = IOBuffer()
    run(; maxiter = 10, discretization = RC(at = 5), verbose = 1, io = buf)
    s = String(take!(buf))
    @test occursin("discretization: RebuildCheck, design and physics rebuilt at iteration 5 (same mesh)", s)
    @test occursin("problem from design_setup/physics_setup", s)
    @test occursin("iteration 5: design and physics rebuilt", s) && occursin("objective at the rebuilt problem: identical", s)
    @test occursin("1 rebuild event", s)
    buf = IOBuffer()
    run(; maxiter = 4, verbose = 1, io = buf)
    @test occursin("discretization: fixed mesh (default)", String(take!(buf)))
    ncalls = Ref(0)
    ds, ps = mbb_factories(; proj = TanhProjection(β = 1.0, η = 0.5), output = false, load = () -> (ncalls[] += 1; 1.0 + 0.1 * (ncalls[] - 1)))
    pb = Optimization(model; design_setup = ds, physics_setup = ps)
    buf = IOBuffer()
    solve!(pb, method(:SiMPL); maxiter = 6, verbose = 1, io = buf, stop = nothing, discretization = RC(at = 3))
    @test occursin(r"objective at the rebuilt problem: differs by [0-9.e-]+ \(relative\)", String(take!(buf)))

    # the contract: due allocates nothing, labels, errors of RebuildCheck
    d = RC(every = 3)
    HyperShape.Methods.due(d, 6, nothing, nothing, nothing)
    @test @allocated(HyperShape.Methods.due(d, 6, nothing, nothing, nothing)) == 0
    @test HyperShape.Methods.due(RC(at = [3, 9]), 9, nothing, nothing, nothing) && !HyperShape.Methods.due(RC(at = 3), 4, nothing, nothing, nothing)
    @test !HyperShape.Methods.needs_setup(FixedMesh()) && HyperShape.Methods.needs_setup(d)
    @test_throws ArgumentError RC()
    @test_throws ArgumentError RC(at = 3, every = 2)
    @test_throws ArgumentError RC(every = 0)
    @test_throws ArgumentError RC(at = 3, what = :mesh)

    # errors when the problem is built
    ds0, ps0 = mbb_factories(; output = false)
    @test_throws ArgumentError Optimization(model; design_setup = m -> (; z = ds0(m).z), physics_setup = ps0)
    @test_throws ArgumentError Optimization(model; design_setup = ds0, physics_setup = (m, x) -> (; C = 1))
    @test_throws ArgumentError Optimization(model; design_setup = ds0,
                                            physics_setup = (m, x) -> (r = ps0(m, x); (; objective = r.objective, monitor = r.monitors)))
    other = mbb_model(24, 8)
    @test_throws ArgumentError Optimization(model; design_setup = m -> ds0(other), physics_setup = ps0)
    dz = ds0(model)
    @test_throws ArgumentError Optimization(model; design_setup = m -> dz, physics_setup = (m, x) -> ps0(m, ds0(m).x))

    # errors of solve!: discretisation without factories, monitors or driver given twice or tied to the old problem,
    # metric tied to the mesh, factories that capture the variables, the functionals or a continued projection
    q = compliance_problem()
    @test_throws ArgumentError solve!(q.problem, method(:SiMPL); maxiter = 3, verbose = 0, discretization = RC(at = 1))
    ds, ps = mbb_factories(; output = true)
    pb = Optimization(model; design_setup = ds, physics_setup = ps)
    grey = Grey(pb.physics.θ, pb.physics.dΩ)
    @test_throws ArgumentError solve!(pb, method(:SiMPL); maxiter = 3, verbose = 0, monitors = (g2 = grey,))
    post0 = OptPostProcessor(every = 1)
    postd = OptPostProcessor((post, ctx) -> nothing; every = 1)
    @test_throws ArgumentError solve!(pb, method(:SiMPL); maxiter = 3, verbose = 0, post = postd)
    dsn, psn = mbb_factories(; output = false, monitors = false)
    pbn = Optimization(model; design_setup = dsn, physics_setup = psn)
    @test_throws ArgumentError solve!(pbn, method(:SiMPL); maxiter = 3, verbose = 0, post = post0)
    @test_throws ArgumentError solve!(pbn, method(:SiMPL); maxiter = 3, verbose = 0, post = postd, discretization = RC(at = 1))
    @test_throws ArgumentError solve!(pbn, method(:SiMPL); maxiter = 3, verbose = 0, monitors = (grey = Grey(pbn.physics.θ, pbn.physics.dΩ),),
                                      discretization = RC(at = 1))
    @test_throws ArgumentError solve!(pbn, DensityMethod(SiMPL(); metric = LumpedMass(pbn.physics.dΩ)); maxiter = 3, verbose = 0,
                                      discretization = RC(at = 1))
    r = solve!(pbn, method(:SiMPL); maxiter = 3, verbose = 0, stop = nothing,
               monitors = (grey = Grey(pbn.physics.θ, pbn.physics.dΩ),))           # without factory monitors, FixedMesh: allowed
    @test haskey(r.history.monitors, :grey)
    dsx, psx = mbb_factories(; output = false)
    dfix = dsx(model)
    pbx = Optimization(model; design_setup = m -> dfix, physics_setup = psx)
    # a design_setup that returns the same design is a rebuild of the physics only (on the same mesh it is harmless;
    # on another mesh its spaces are not on the new model, an error when the problem is built)
    rx = solve!(pbx, method(:SiMPL); maxiter = 4, verbose = 0, stop = nothing, discretization = RC(at = 2))
    @test rx.problem.z === pbx.z
    dsy, psy = mbb_factories(; output = false)
    pfix = Ref{Any}(nothing)
    psfix = (m, x) -> (pfix[] === nothing && (pfix[] = psy(m, x)); pfix[])
    pby = Optimization(model; design_setup = dsy, physics_setup = psfix)
    @test_throws r"same objective" solve!(pby, method(:SiMPL); maxiter = 4, verbose = 0, stop = nothing,
                                          discretization = RC(at = 2, what = :physics))
    proj = TanhProjection(β = 1.0, η = 0.5)
    ncalls = Ref(0)
    psp = (m, x) -> (ncalls[] += 1; mbb_factories(; output = false, proj = ncalls[] == 1 ? proj : TanhProjection(β = 1.0, η = 0.5))[2](m, x))
    pbp = Optimization(model; design_setup = mbb_factories()[1], physics_setup = psp)
    @test_throws r"TanhProjection continued as β" solve!(pbp, method(:SiMPL); maxiter = 4, verbose = 0, stop = nothing,
                                                         continuation = Continuation(β = proj => [1.0, 2.0]; every = 2),
                                                         discretization = RC(at = 2))
    ncalls[] = 0
    psc = (m, x) -> (ncalls[] += 1; mbb_factories(; output = false, al = ncalls[] > 1)[2](m, x))
    pbc = Optimization(model; design_setup = mbb_factories()[1], physics_setup = psc)
    @test_throws r"the constraints changed" solve!(pbc, method(:SiMPL); maxiter = 4, verbose = 0, stop = nothing,
                                                   discretization = RC(at = 2))

    # output: the same files and the same collection with and without rebuilds; a checkpoint after a rebuild resumes
    function out_run(d, dir; maxiter = 12, output = true, kw...)
      proj = TanhProjection(β = 1.0, η = 0.5)
      ds, ps = mbb_factories(; proj, output)
      pb = Optimization(model; design_setup = ds, physics_setup = ps)
      post = output ? OptPostProcessor(every = 2, filepath = joinpath(dir, "vtk"), async = false) : nothing
      r = solve!(pb, method(:SiMPL); maxiter, verbose = 0, stop = nothing, post, discretization = d,
                 continuation = Continuation(β = proj => [1.0, 2.0, 4.0]; every = 8),
                 checkpoint = Checkpoint(filepath = joinpath(dir, "ck"), every = 3), kw...)
      pb, r
    end
    da, db = mktempdir(), mktempdir()
    _, ra = out_run(FixedMesh(), da)
    _, rb = out_run(RC(every = 4), db)
    @test same_history(ra.history, rb.history)
    fa, fb = sort(readdir(joinpath(da, "vtk"))), sort(readdir(joinpath(db, "vtk")))
    @test fa == fb && "optimization.pvd" in fa && length(fa) >= 6
    @test read(joinpath(da, "vtk", "optimization.pvd"), String) == read(joinpath(db, "vtk", "optimization.pvd"), String)
    for f in fa
      endswith(f, ".vtu") && @test read(joinpath(da, "vtk", f)) == read(joinpath(db, "vtk", f))
    end
    ck = load_checkpoint(joinpath(db, "ck"))
    @test ck.discretization == "RebuildCheck" && ck.events == 2 && ck.iteration == 12
    # a checkpoint written right after a rebuild (iteration 9 of every = 3, a rebuild at 8): resume it with FixedMesh
    dc = mktempdir()
    out_run(RC(at = 8), dc; maxiter = 9)
    ck = load_checkpoint(joinpath(dc, "ck"))
    @test ck.events == 1 && ck.iteration == 9 && ck.memory !== nothing
    dd = mktempdir()
    _, rd = out_run(FixedMesh(), dd; output = false, resume = joinpath(dc, "ck"))
    @test same_history(rd.history, ra.history) && get_values(rd.z) == get_values(ra.z)
    @test load_checkpoint(joinpath(dd, "ck")).events == 1
    # an exception in a factory at a rebuild: checkpoint from the state of the event (with the memory), then rethrown
    ncalls[] = 0
    psb = (m, x) -> (ncalls[] += 1; ncalls[] > 1 && error("boom in physics_setup"); mbb_factories(; output = false)[2](m, x))
    pbb = Optimization(model; design_setup = mbb_factories()[1], physics_setup = psb)
    de = mktempdir()
    @test_throws ErrorException solve!(pbb, method(:SiMPL); maxiter = 6, verbose = 0, stop = nothing,
                                       discretization = RC(at = 4), checkpoint = Checkpoint(filepath = de, every = 5))
    cke = load_checkpoint(de)
    @test cke.iteration == 4 && cke.memory !== nothing
    # the same when the check after the rebuild fails (factories that change the constraints)
    ncalls[] = 0
    pbc = Optimization(model; design_setup = mbb_factories()[1], physics_setup = psc)
    dg = mktempdir()
    @test_throws r"the constraints changed" solve!(pbc, method(:SiMPL); maxiter = 6, verbose = 0, stop = nothing,
                                                   discretization = RC(at = 4), checkpoint = Checkpoint(filepath = dg, every = 3))
    ckg = load_checkpoint(dg)
    @test ckg.iteration == 4 && ckg.memory !== nothing && pbc.state[].iteration == 4

    # review fixes: the scaled start survives rebuilds; an output with typed arguments; a discretisation of the user
    # that keeps the design (rebuild(…; what = :physics)); OC with its move continued; a resume at an iteration of a rebuild
    function scaled(d)
      ds, ps = mbb_factories(; linear = true, output = false)
      pb = Optimization(model; design_setup = ds, physics_setup = ps)
      set_values!(pb.z, fill(0.8, length(pb.z)))
      solve!(pb, method(:SiMPL); maxiter = 6, verbose = 0, stop = nothing, discretization = d)
    end
    r0, r1 = scaled(FixedMesh()), scaled(RC(at = 3))
    @test r0.start_scaling !== nothing && r1.start_scaling == r0.start_scaling && same_history(r0.history, r1.history)
    dsT, psT = mbb_factories(; output = false)
    psTyped = (m, x) -> (r = psT(m, x); (; r..., output = (post::OptPostProcessor, ctx) -> nothing))
    @test Optimization(model; design_setup = dsT, physics_setup = psTyped) isa Optimization
    @test_throws ArgumentError Optimization(model; design_setup = dsT, physics_setup = (m, x) -> (; psT(m, x)..., output = 1))
    @test_throws ArgumentError Optimization(model; design_setup = dsT)
    @test_throws ArgumentError Optimization(pb.design.x; objective = pb.objective)
    _, ref = run(; maxiter = 10)
    for what in (:physics, :design)
      buf = IOBuffer()
      pbk, rk = run(; maxiter = 10, discretization = _KeepDesign(4, what), verbose = 1, io = buf)
      s = String(take!(buf))
      @test same_history(rk.history, ref.history) && (rk.problem.z === pbk.z) == (what === :physics)
      @test occursin("discretization: _KeepDesign", s) && occursin("iteration 4: " * (what === :physics ? "physics" : "design and physics") * " rebuilt", s)
    end
    @test_throws ArgumentError HyperShape.Methods.rebuild(pb; what = :physics, model = mbb_model(24, 8))
    pbm = HyperShape.Methods.rebuild(pb; model = mbb_model(24, 8))
    @test pbm.setup.model !== pb.setup.model && pbm.state === pb.state
    function oc_run(d)
      proj = TanhProjection(β = 1.0, η = 0.5)
      ds, ps = mbb_factories(; proj, output = false)
      pb = Optimization(model; design_setup = ds, physics_setup = ps)
      opt = OC(move = 0.2)
      r = solve!(pb, DensityMethod(opt; metric = LumpedMass()); maxiter = 12, verbose = 0, stop = nothing, discretization = d,
                 continuation = Continuation(β = proj => [1.0, 2.0, 4.0], move = opt => [0.2, 0.1, 0.05]; every = 4))
      r, opt
    end
    (ra, oa), (rb, ob) = oc_run(FixedMesh()), oc_run(RC(every = 3))
    @test same_history(ra.history, rb.history) && get_values(ra.z) == get_values(rb.z) && oa.move == ob.move == 0.2
    dr = mktempdir()
    pbr, _ = run(; maxiter = 8, discretization = RC(every = 4), checkpoint = Checkpoint(filepath = dr, every = 4))
    @test pbr.state[].events == 1                   # at 4; none at 8, the last iteration
    pbs, rs = run(; maxiter = 16, discretization = RC(every = 4), resume = dr)
    _, rf = run(; maxiter = 16, discretization = RC(every = 4))
    @test same_history(rs.history, rf.history) && get_values(rs.z) == get_values(rf.z)
    @test pbs.state[].events == 2 && rf.problem.state[].events == 3   # resumed at 8: no rebuild at 8 (4 and 12 only)
    buf = IOBuffer()
    run(; maxiter = 4, discretization = RC(at = 9), verbose = 1, io = buf)
    @test occursin("no rebuild before maxiter", String(take!(buf)))
  end
end
