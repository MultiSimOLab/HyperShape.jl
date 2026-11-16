using Gridap
using Gridap.FESpaces
using HyperFEM
using HyperFEMKernels
using HyperShape
using HyperShape.Variables: update_fields!, set_values!, get_values, changed!, set_beta!, version
using HyperShape.Operators: apply!, get_model
using Base.CoreLogging: with_logger, SimpleLogger, Error as LogError
using LinearAlgebra
using Test

function taylor_rates(value, g, z, d)
  z0 = copy(get_values(z))
  J0 = value()
  gd = dot(g, d)
  errs = Float64[]
  for h in (1e-2, 5e-3, 2.5e-3, 1.25e-3)
    set_values!(z, clamp.(z0 .+ h .* d, 0.0, 1.0))
    push!(errs, abs(value() - J0 - h * gd))
  end
  set_values!(z, z0)
  [log2(errs[i] / errs[i+1]) for i in 1:length(errs)-1]
end

direction(n) = [0.1 * sin(1.7i) for i in 1:n]

struct _NoSpaces <: HyperFEM.ComputationalModels.ComputationalModel end   # a model that does not expose its spaces

@testset "Functionals" begin

  @testset "Compliance, self-adjoint ($(design))" for design in (:P0, :P1)
    p = state_problem(; design)
    (; z, θ, S, f, dk, dΩ) = p
    J = Functional((u, θ) -> ∫(f * u)dΩ, S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = :self, fields = θ, scale = 1.0)
    g = functional_gradient!(zeros(length(z)), J, nothing)
    rates = taylor_rates(() -> functional_value(J, nothing), g, z, direction(length(z)))
    @test all(r -> 1.9 < r < 2.1, rates)
  end

  @testset "Boundary value with its own adjoint ($(design))" for design in (:P0, :P1)
    p = state_problem(; design)
    (; z, θ, S, A, dk, dΩ, dΓ) = p
    J = Functional((u, θ) -> ∫(u)dΓ, S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = A, fields = θ, scale = 1.0)
    g = functional_gradient!(zeros(length(z)), J, nothing)
    rates = taylor_rates(() -> functional_value(J, nothing), g, z, direction(length(z)))
    @test all(r -> 1.9 < r < 2.1, rates)
  end

  @testset "Design-only functional (volume of θ)" begin
    p = state_problem()
    (; z, θ, dΩ) = p
    Vol = Functional(θ -> ∫(θ.ρ)dΩ, θ; derivative = θ -> (δ -> ∫(δ)dΩ), scale = 0.5)
    @test functional_value(Vol, nothing) ≈ sum(∫(θ.ρ)dΩ) / 0.5
    g = functional_gradient!(zeros(length(z)), Vol, nothing)
    rates = taylor_rates(() -> functional_value(Vol, nothing), g, z, direction(length(z)))
    @test all(r -> 1.9 < r < 2.1, rates)
  end

  @testset "Scale" begin
    p = state_problem()
    (; z, θ, S, f, dk, dΩ) = p
    Jf, DJf = (u, θ) -> ∫(f * u)dΩ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ)
    J1 = Functional(Jf, S; derivative = DJf, adjoint = :self, fields = θ, scale = 1.0)
    Ji = Functional(Jf, S; derivative = DJf, adjoint = :self, fields = θ, scale = :initial)
    J0 = functional_value(J1, nothing)
    @test functional_value(Ji, nothing) ≈ 1.0
    g1 = functional_gradient!(zeros(length(z)), J1, nothing)
    gi = functional_gradient!(zeros(length(z)), Ji, nothing)
    @test gi ≈ g1 ./ J0
    w = copy(get_values(z)); w .= clamp.(w .+ 0.05, 0, 1)
    set_values!(z, w)
    @test functional_value(Ji, nothing) ≈ functional_value(J1, nothing) / J0   # the scale stays the first value
    # gradient first: the scale is set from the value at that design
    Jg = Functional(Jf, S; derivative = DJf, adjoint = :self, fields = θ, scale = :initial)
    gg = functional_gradient!(zeros(length(z)), Jg, nothing)
    @test functional_value(Jg, nothing) ≈ 1.0
    @test gg ≈ functional_gradient!(zeros(length(z)), J1, nothing) ./ functional_value(J1, nothing)
    for bad in (0.0, -1.0, NaN, Inf, :first)
      @test_throws ArgumentError Functional(Jf, S; derivative = DJf, adjoint = :self, fields = θ, scale = bad)
    end
    @test_throws UndefKeywordError Functional(Jf, S; derivative = DJf, adjoint = :self, fields = θ)
  end

  @testset "States are solved only when something changed" begin
    p = state_problem()
    (; z, θ, H, S, A, f, dk, dΩ, dΓ, nforward, nadjoint) = p
    C = Functional((u, θ) -> ∫(f * u)dΩ, S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = :self, fields = θ, scale = 1.0)
    B = Functional((u, θ) -> ∫(u)dΓ, S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = A, fields = θ, scale = 1.0)
    g = zeros(length(z))
    functional_value(C, nothing); functional_value(B, nothing)
    functional_gradient!(g, C, nothing); functional_gradient!(g, B, nothing)
    @test (nforward[], nadjoint[]) == (1, 1)              # one forward solve shared, one adjoint
    vS = version(S)
    functional_gradient!(g, B, nothing)
    @test (nforward[], nadjoint[]) == (1, 1) && version(S) == vS
    changed!(z)
    functional_value(C, nothing); functional_gradient!(g, B, nothing)
    @test (nforward[], nadjoint[]) == (2, 2)
    set_beta!(H, 6.0)                                      # a new β changes θ: states solved again
    functional_value(C, nothing)
    @test nforward[] == 3
  end

  @testset "Failed forward solve" begin
    p = state_problem()
    (; z, θ, f, dk, dΩ) = p
    converge = Ref(false)
    Sf = StateOperator(get_model(p.S); depends = (θ,),
                       solve = (m, ctx) -> (solve!(m; Assembly = true), converge[]))
    J = Functional((u, θ) -> ∫(f * u)dΩ, Sf; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = :self, fields = θ, scale = 1.0)
    @test isnan(functional_value(J, nothing))
    g = functional_gradient!(zeros(length(z)), J, nothing)
    @test all(isnan, g)
    converge[] = true
    @test isnan(functional_value(J, nothing))              # the failed solve is not repeated for the same design
    changed!(z)
    @test isfinite(functional_value(J, nothing))
    @test all(isfinite, functional_gradient!(g, J, nothing))
  end

  @testset "Several physical fields: DJform as a named tuple" begin
    p = state_problem()
    (; z, x, f, dk, dΩ) = p
    θ2 = PhysicalFields(x; ρ = Identity(), r2 = :ρ => Heaviside(β = 2.0, η = 0.5))
    J = Functional(θ -> ∫(θ.ρ * θ.r2)dΩ, θ2; derivative = θ -> (ρ = δ -> ∫(θ.r2 * δ)dΩ, r2 = δ -> ∫(θ.ρ * δ)dΩ), scale = 1.0)
    g = functional_gradient!(zeros(length(z)), J, nothing)
    rates = taylor_rates(() -> functional_value(J, nothing), g, z, direction(length(z)))
    @test all(r -> 1.9 < r < 2.1, rates)
    Jbad = Functional(θ -> ∫(θ.ρ)dΩ, θ2; derivative = θ -> (δ -> ∫(δ)dΩ), scale = 1.0)
    @test_throws ArgumentError functional_gradient!(zeros(length(z)), Jbad, nothing)
  end

  @testset "StateOperator: contract" begin
    p = state_problem()
    m = HyperShape.Operators.get_model(p.S)
    @test_throws ArgumentError StateOperator(m; depends = (1.0,), solve = (m, ctx) -> solve!(m))
    @test_throws ArgumentError StateOperator(m; solve = (m, ctx) -> solve!(m))     # dependencies are mandatory
    @test StateOperator(m; depends = (p.θ,)).solve === HyperShape.Operators.default_solve(StateOperator, m)
    # forms that captured the state see the new solution after a new design
    uh = get_state(p.S)
    dΩ = p.dΩ
    apply!(p.S, nothing)
    w = clamp.(get_values(p.z) .+ 0.1, 0, 1)
    set_values!(p.z, w)
    apply!(p.S, nothing)
    fresh = solve(LUSolver(), AffineFEOperator(p.a, p.l, p.U, p.V))
    @test sum(∫(uh * uh)dΩ) ≈ sum(∫(fresh * fresh)dΩ) rtol = 1e-12
    # depends as one object; a solve that returns a non-Bool flag
    @test StateOperator(m; depends = p.θ, solve = (m, ctx) -> solve!(m)).depends == (p.θ,)
    S1 = StateOperator(m; depends = p.θ, solve = (m, ctx) -> (solve!(m; Assembly = true), 1))
    @test apply!(S1, nothing)                                # HyperFEM's flagconv = 1
    S0 = StateOperator(m; depends = p.θ, solve = (m, ctx) -> (solve!(m; Assembly = true), 0))
    @test !apply!(S0, nothing)
    Sbad = StateOperator(m; depends = p.θ, solve = (m, ctx) -> (solve!(m; Assembly = true), 2))
    @test_throws ArgumentError apply!(Sbad, nothing)
    ctxs = Any[]
    S2 = StateOperator(m; depends = (p.θ,), solve = (m, ctx) -> (push!(ctxs, ctx); solve!(m; Assembly = true)))
    @test apply!(S2, (iteration = 3,)) && ctxs == [(iteration = 3,)]
  end

  @testset "Non-zero Dirichlet values: adjoint on the homogeneous space" begin
    p = state_problem(; gD = 1.0)
    (; z, θ, S, A, dk, dΩ, dΓ) = p
    J = Functional((u, θ) -> ∫(u)dΓ, S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = A, fields = θ, scale = 1.0)
    g = functional_gradient!(zeros(length(z)), J, nothing)
    @test all(r -> 1.9 < r < 2.1, taylor_rates(() -> functional_value(J, nothing), g, z, direction(length(z))))
    # adjoint built on the space with the Dirichlet values of the state: detected
    q = state_problem(; gD = 1.0, adjoint_space = :state)
    @test_throws ArgumentError Functional((u, θ) -> ∫(u)dΓ, q.S; adjoint = q.A, scale = 1.0,
                                          derivative = (u, w, θ) -> (δ -> ∫(-(q.dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)q.dΩ))
    # self-adjoint shortcut with non-zero Dirichlet values: detected when the functional is built
    err = try
      Functional((u, θ) -> ∫(p.f * u)dΩ, S; adjoint = :self, scale = 1.0,
                 derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ))
      nothing
    catch e
      e
    end
    @test err isa ArgumentError && occursin("adjoint = :self", err.msg)
  end

  @testset "Adjoint of a failed state is not solved" begin
    p = state_problem()
    (; θ, dΓ, dk, dΩ) = p
    converge = Ref(false)
    nadj = Ref(0)
    Sf = StateOperator(get_model(p.S); depends = θ, solve = (m, ctx) -> (solve!(m; Assembly = true), converge[]))
    Af = StateOperator(get_model(p.A); depends = (θ, Sf), solve = (m, ctx) -> (nadj[] += 1; solve!(m; Assembly = true)))
    @test !apply!(Af, nothing) && !is_converged(Af) && !is_converged(Sf) && nadj[] == 0
    converge[] = true
    changed!(p.z)
    @test apply!(Af, nothing) && is_converged(Af) && nadj[] == 1
  end

  @testset "Dependencies declared in the functional" begin
    p = state_problem()
    (; θ, dΓ, dΩ, dk) = p
    m = get_model(p.S)
    S0 = StateOperator(m; depends = (), solve = (m, ctx) -> solve!(m; Assembly = true))
    A1 = StateOperator(get_model(p.A); depends = θ, solve = (m, ctx) -> solve!(m; Assembly = true))
    Jf, DJf = (u, θ) -> ∫(u)dΓ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ)
    @test_throws ArgumentError Functional(Jf, S0; derivative = DJf, adjoint = p.A, fields = θ, scale = 1.0)
    @test_throws ArgumentError Functional(Jf, p.S; derivative = DJf, adjoint = A1, fields = θ, scale = 1.0)
  end

  @testset "Value kept until the design changes; ctx reaches every solve" begin
    p = state_problem()
    (; z, θ, S, A, dΓ, dk, dΩ, ctxs) = p
    nJ = Ref(0)
    J = Functional((u, θ) -> (nJ[] += 1; ∫(u)dΓ), S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = A, fields = θ, scale = 1.0)
    v1 = functional_value(J, nothing); v2 = functional_value(J, nothing)
    @test v1 == v2 && nJ[] == 1
    changed!(z)
    functional_value(J, nothing)
    @test nJ[] == 2
    empty!(ctxs)
    changed!(z)
    ctx = (iteration = 7,)
    functional_gradient!(zeros(length(z)), J, ctx)
    @test (:S, ctx) in ctxs && (:A, ctx) in ctxs
    # no allocation when nothing changed
    apply!(S, nothing); apply!(A, nothing)
    @test (@allocated apply!(S, nothing)) == 0
    @test (@allocated apply!(A, nothing)) == 0
  end

  @testset "Taylor test after a change of β" begin
    p = state_problem()
    (; z, θ, H, S, A, dk, dΩ, dΓ, nadjoint) = p
    J = Functional((u, θ) -> ∫(u)dΓ, S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = A, fields = θ, scale = 1.0)
    functional_gradient!(zeros(length(z)), J, nothing)
    n = nadjoint[]
    set_beta!(H, 8.0)
    g = functional_gradient!(zeros(length(z)), J, nothing)
    @test nadjoint[] == n + 1
    @test all(r -> 1.9 < r < 2.1, taylor_rates(() -> functional_value(J, nothing), g, z, direction(length(z))))
  end

  @testset "Nonlinear state with a load ramp" begin
    model = CartesianDiscreteModel((0, 1, 0, 1), (8, 8))
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "left", [1, 3, 7]); add_tag_from_tags!(labels, "right", [2, 4, 8])
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)
    dΓ = Measure(BoundaryTriangulation(model; tags = "right"), 2)
    Vρ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
    ρ0 = [0.3 + 0.5 * abs(sin(2.3i)) for i in 1:num_free_dofs(Vρ)]
    z, x, θ = single_field_design(:ρ, Vρ; bounds = (0.0, 1.0), init = ρ0, physical = Identity())
    k(ρ) = 1e-2 + ρ^3; dk(ρ) = 3ρ^2
    c = 5.0
    bc = BoundaryConditions(BoundaryCondition("left", 1.0, Λ -> Λ))
    bc0 = BoundaryConditions(BoundaryCondition("left", 0.0))
    V = TestFESpace(Ω, ReferenceFE(lagrangian, Float64, 1), bc, conformity = :H1)
    U = TrialFESpace(V, bc, 1.0); U0 = TrialFESpace(V, bc0, 1.0)
    res(Λ) = (u, v) -> ∫((k ∘ θ.ρ) * (1 + c * u * u) * ∇(u) ⋅ ∇(v) - Λ * v)dΩ
    jac(Λ) = (u, du, v) -> ∫((k ∘ θ.ρ) * ((1 + c * u * u) * ∇(du) ⋅ ∇(v) + 2c * u * du * ∇(u) ⋅ ∇(v)))dΩ
    function build(maxiter, stepping)
      nls = NewtonRaphson(LUSolver(); maxiter, atol = 1e-11, rtol = 1e-11, verbose = 0)
      S = StateOperator(StaticNonlinearModel(res, jac, U, V, bc; nls), θ;
                        solve = (m, ctx) -> solve!(m; stepping = stepping[], RestartState = false))
      uh = get_state(S)
      A = AdjointOperator(S, StaticLinearModel(w -> ∫(w)dΓ, (p, w) -> jac(1.0)(uh, w, p), U0, V, bc0; ls = LUSolver()))
      @test A.depends === (θ, S)
      Functional((u, θ) -> ∫(u)dΓ, S; adjoint = A, scale = 1.0,
                 derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * (1 + c * u * u) * ∇(u) ⋅ ∇(w) * δ)dΩ)), S
    end
    with_logger(SimpleLogger(stderr, LogError)) do
      J, S = build(30, Ref{Any}((nsteps = 2, maxbisec = 5)))
      g = functional_gradient!(zeros(length(z)), J, nothing)
      @test all(r -> 1.9 < r < 2.1, taylor_rates(() -> functional_value(J, nothing), g, z, direction(length(z))))
      # Newton fails (few iterations, no bisection): NaN, then a converged solve at the next design
      stepping = Ref{Any}((nsteps = 2, maxbisec = 0))
      Jf, Sf = build(3, stepping)
      changed!(z)
      @test isnan(functional_value(Jf, nothing)) && !is_converged(Sf)
      @test all(isnan, functional_gradient!(zeros(length(z)), Jf, nothing))
      stepping[] = (nsteps = 60, maxbisec = 10)
      Jok, _ = build(30, Ref{Any}((nsteps = 2, maxbisec = 5)))
      changed!(z)
      @test functional_value(Jf, nothing) ≈ functional_value(Jok, nothing) rtol = 1e-8
    end
  end

  @testset "Default solve, dependencies, AdjointOperator and the keyword form of Functional" begin
    p = state_problem()
    (; z, θ, S, A, f, dk, dΩ, dΓ, a, l, U, U0, V, bc, bc0) = p
    # default solve of a StaticLinearModel: the same states as the explicit solve
    Sd = StateOperator(StaticLinearModel(l, a, U, V, bc; ls = LUSolver()), θ)
    Ad = AdjointOperator(Sd, StaticLinearModel(w -> ∫(w)dΓ, (p, w) -> a(w, p), U0, V, bc0; ls = LUSolver()))
    @test Sd.solve === HyperShape.Operators.default_solve(StateOperator, Sd.model)
    @test Ad.depends === (θ, Sd)
    Jd = Functional((u, θ) -> ∫(u)dΓ, Sd; adjoint = Ad, scale = 1.0,
                    derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ))
    Je = Functional((u, θ) -> ∫(u)dΓ, S; adjoint = A, scale = 1.0, fields = θ,
                    derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ))
    @test Jd.θ === θ                                   # θ deduced from the dependencies of S
    @test functional_value(Jd, nothing) == functional_value(Je, nothing)
    @test functional_gradient!(zeros(length(z)), Jd, nothing) ≈ functional_gradient!(zeros(length(z)), Je, nothing) rtol = 1e-12
    @test get_free_dof_values(get_state(Sd)) == get_free_dof_values(get_state(S))
    @test get_free_dof_values(get_state(Ad)) == get_free_dof_values(get_state(A))
    # default solve of the filter
    Lf = LinearOperator(StaticLinearModel((u, v) -> ∫(u * v)dΩ, U0, V, NothingBC(); ls = LUSolver()))
    @test Lf.depends === () && Lf.solve === HyperShape.Operators.default_solve(LinearOperator, Lf.model)
    # dependencies: positionally or with depends =, once; mandatory for a state
    @test StateOperator(Sd.model; depends = θ).depends === (θ,)
    @test_throws ArgumentError StateOperator(Sd.model, θ; depends = θ)
    @test_throws ArgumentError StateOperator(Sd.model)
    @test_throws ArgumentError LinearOperator(Lf.model, (); depends = ())
    @test LinearOperator(Lf.model, θ).depends === (θ,)
    # a nonlinear model has no default solve
    res = (u, v) -> ∫((1 + u * u) * ∇(u) ⋅ ∇(v) - v)dΩ
    jac = (u, du, v) -> ∫((1 + u * u) * ∇(du) ⋅ ∇(v) + 2 * u * du * ∇(u) ⋅ ∇(v))dΩ
    nl = StaticNonlinearModel(Λ -> res, Λ -> jac, U, V, bc; nls = NewtonRaphson(LUSolver(); verbose = 0))
    err = try StateOperator(nl, θ); nothing catch e; e end
    @test err isa ArgumentError && occursin("StaticNonlinearModel", err.msg) && occursin("stepping", err.msg)
    @test StateOperator(nl, θ; solve = (m, ctx) -> solve!(m)).depends === (θ,)
    # AdjointOperator: extra dependencies without repetitions, homogeneous trial space
    Ae = AdjointOperator(Sd, Ad.model, (θ, Sd, z); solve = (m, ctx) -> solve!(m; Assembly = true))
    @test Ae.depends === (θ, Sd, z)
    bc1 = BoundaryConditions(BoundaryCondition("left", 1.0))
    U1 = TrialFESpace(V, bc1, 1.0)
    @test_throws ArgumentError AdjointOperator(Sd, StaticLinearModel(w -> ∫(w)dΓ, (p, w) -> a(w, p), U1, V, bc1;
                                                                    ls = LUSolver()))
    # Functional: adjoint mandatory, :self or a state; fields deduced or given
    DJ = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ)
    err = try Functional((u, θ) -> ∫(u)dΓ, Sd; derivative = DJ, scale = 1.0); nothing catch e; e end
    @test err isa ArgumentError && occursin("AdjointOperator", err.msg) && occursin(":self", err.msg)
    @test_throws ArgumentError Functional((u, θ) -> ∫(u)dΓ, Sd; derivative = DJ, adjoint = Sd, scale = 1.0)
    @test_throws ArgumentError Functional((u, θ) -> ∫(u)dΓ, Sd; derivative = DJ, adjoint = :none, scale = 1.0)
    @test_throws ArgumentError Functional((u, θ) -> ∫(u)dΓ, Sd; derivative = DJ, adjoint = S, scale = 1.0)  # not on Sd
    θ2 = PhysicalFields(p.x; ρ = Identity())
    S2 = StateOperator(Sd.model, (θ, θ2))
    @test_throws ArgumentError Functional((u, θ) -> ∫(u)dΓ, S2; derivative = DJ, adjoint = :self, scale = 1.0)
    @test Functional((u, θ) -> ∫(u)dΓ, S2; derivative = DJ, adjoint = :self, scale = 1.0, fields = θ).θ === θ
    @test_throws ArgumentError Functional((u, θ) -> ∫(u)dΓ, StateOperator(Sd.model, z); derivative = DJ,
                                          adjoint = :self, scale = 1.0)
    @test_throws ArgumentError Functional((u, θ) -> ∫(u)dΓ, Sd; derivative = DJ, adjoint = :self, scale = 1.0,
                                          fields = θ2)                              # Sd does not depend on θ2
    # repeated dependencies are kept once; a state reached only through another state cannot give θ
    @test StateOperator(Sd.model, (θ, θ)).depends === (θ,)
    @test Functional((u, θ) -> ∫(u)dΓ, StateOperator(Sd.model, (θ, θ)); derivative = DJ, adjoint = :self,
                     scale = 1.0).θ === θ
    err = try Functional((u, θ) -> ∫(u)dΓ, StateOperator(Sd.model, Sd); derivative = DJ, adjoint = :self,
                         scale = 1.0); nothing catch e; e end
    @test err isa ArgumentError && occursin("StateOperator(model, θ)", err.msg)
    # default_solve with the concrete type of an operator; several right-hand sides have none
    @test HyperShape.Operators.default_solve(typeof(Sd), Sd.model) === Sd.solve
    @test HyperShape.Operators.default_solve(typeof(Lf), Lf.model) === Lf.solve
    multi = StaticLinearModel([v -> ∫(v)dΩ, v -> ∫(2v)dΩ], a, U0, V, bc0; ls = LUSolver())
    @test_throws ArgumentError StateOperator(multi, θ)
    @test_throws ArgumentError LinearOperator(multi)
    # Dirichlet values of a multi-field trial space are read; a model without spaces is not checked
    multi1 = StaticLinearModel([v -> ∫(v)dΩ, v -> ∫(2v)dΩ], a, U1, V, bc1; ls = LUSolver())
    vals = HyperShape.Operators.dirichlet_values(multi1)
    @test vals !== nothing && any(!iszero, vals)
    @test_throws ArgumentError AdjointOperator(Sd, multi1; solve = (m, ctx) -> solve!(m))
    # an adjoint of a state that depends on another state warns that its sensitivity is not included
    Sc = StateOperator(Sd.model, (θ, Sd))
    @test_logs (:warn, r"depends on another state") AdjointOperator(Sc, Ad.model)
    @test_logs AdjointOperator(Sd, Ad.model)
    @test HyperShape.Operators.dirichlet_values(_NoSpaces()) === nothing
  end

  @testset "Lazy and compiled backends after changing z and β" begin
    pl = state_problem(; backend = :lazy)
    pc = state_problem(; backend = :compiled)
    funcs = map((pl, pc)) do p
      dΓ, dΩ, dk = p.dΓ, p.dΩ, p.dk
      Functional((u, θ) -> ∫(u)dΓ, p.S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = p.A, fields = p.θ, scale = 1.0)
    end
    for p in (pl, pc)
      functional_gradient!(zeros(length(p.z)), funcs[p === pl ? 1 : 2], nothing)
      set_values!(p.z, clamp.(get_values(p.z) .+ 0.07, 0, 1))
      set_beta!(p.H, 7.0)
    end
    gl = functional_gradient!(zeros(length(pl.z)), funcs[1], nothing)
    gc = functional_gradient!(zeros(length(pc.z)), funcs[2], nothing)
    @test maximum(abs, gl .- gc) <= 1e-10 * maximum(abs, gl)
    @test functional_value(funcs[1], nothing) ≈ functional_value(funcs[2], nothing) rtol = 1e-12
  end

  @testset "Two functionals alternate through the same filter (compiled and lazy)" begin
    grads = map((:compiled, :lazy)) do vb
      p = state_problem(; vector_backend = vb)
      (; z, θ, S, A, f, dk, dΩ, dΓ) = p
      C = Functional((u, θ) -> ∫(f * u)dΩ, S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = :self, fields = θ, scale = 1.0)
      B = Functional((u, θ) -> ∫(u)dΓ, S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = A, fields = θ, scale = 1.0)
      Vol = Functional(θ -> ∫(θ.ρ)dΩ, θ; derivative = θ -> (δ -> ∫(δ)dΩ), scale = 1.0)
      gs = Vector{Float64}[]
      for it in 1:3
        for F in (C, B, Vol)
          push!(gs, functional_gradient!(zeros(length(z)), F, nothing))
        end
        set_values!(z, clamp.(get_values(z) .+ 0.03, 0, 1))
      end
      gs
    end
    @test all(k -> maximum(abs, grads[1][k] .- grads[2][k]) <= 1e-10 * maximum(abs, grads[2][k]), eachindex(grads[1]))
  end

  @testset "Exported names do not clash with Gridap and HyperFEM" begin
    M = Module()
    Core.eval(M, :(using Gridap, HyperFEM, HyperShape))
    for name in names(HyperShape)
      name === :HyperShape && continue
      @test Core.eval(M, name) === getfield(HyperShape, name)
    end
  end

end
