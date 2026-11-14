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

# Heat-conduction-like problem: -∇⋅(k(θ.ρ) ∇u) = f in Ω, u = 0 on the left side, with k(ρ) = kmin + ρ³.
function state_problem(; n = 8, design = :P0, backend = :lazy, gD = 0.0, adjoint_space = :homogeneous)
  model = CartesianDiscreteModel((0, 1, 0, 1), (n, n))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "left", [1, 3, 7])
  add_tag_from_tags!(labels, "right", [2, 4, 8])
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  Γ = BoundaryTriangulation(model; tags = "right")
  dΓ = Measure(Γ, 2)
  order = design === :P0 ? 0 : 1
  Vρ = TestFESpace(model, ReferenceFE(lagrangian, Float64, order); conformity = design === :P0 ? :L2 : :H1)
  Vf = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
  Uf = TrialFESpace(Vf)
  af(u, v) = ∫(0.01 * ∇(u) ⋅ ∇(v) + u * v)dΩ
  Sf = LinearOperator(StaticLinearModel(af, Uf, Vf, NothingBC(); ls = LUSolver()); depends = (),
                      solve = (m, b, ctx, stale) -> solve!(m, b; Assembly = stale))
  ρ0 = [0.3 + 0.5 * abs(sin(2.3i)) for i in 1:num_free_dofs(Vρ)]
  H = Heaviside(β = 3.0, η = 0.5)
  z, x, θ = single_field_design(:ρ, Vρ; bounds = (0.0, 1.0), init = ρ0, physical = H ∘ HelmholtzFilter(Sf; measure = dΩ))

  kmin = 1e-2
  k(ρ) = kmin + ρ^3
  dk(ρ) = 3ρ^2
  f(x) = 1 + x[1]
  bc = BoundaryConditions(BoundaryCondition("left", gD))
  bc0 = BoundaryConditions(BoundaryCondition("left", 0.0))
  V = TestFESpace(Ω, ReferenceFE(lagrangian, Float64, 1), bc, conformity = :H1)
  U = TrialFESpace(V, bc, 1.0)
  U0 = TrialFESpace(V, bc0, 1.0)
  a(u, v) = ∫((k ∘ θ.ρ) * ∇(u) ⋅ ∇(v))dΩ
  l(v) = ∫(f * v)dΩ
  nforward, nadjoint = Ref(0), Ref(0)
  ctxs = Any[]
  model_u = StaticLinearModel(l, a, U, V, bc; ls = LUSolver(), backend)
  S = StateOperator(model_u; depends = θ, solve = (m, ctx) -> (nforward[] += 1; push!(ctxs, (:S, ctx)); solve!(m; Assembly = true)))
  uh = get_state(S)
  # adjoint of J = ∫_Γ u dΓ: a(w, p) = ∫_Γ w dΓ for all w (a is symmetric), on the homogeneous space
  Ua, bca = adjoint_space === :homogeneous ? (U0, bc0) : (U, bc)
  model_p = StaticLinearModel(w -> ∫(w)dΓ, (p, w) -> a(w, p), Ua, V, bca; ls = LUSolver(), backend)
  A = StateOperator(model_p; depends = (θ, S), solve = (m, ctx) -> (nadjoint[] += 1; push!(ctxs, (:A, ctx)); solve!(m; Assembly = true)))
  (; z, x, θ, H, S, A, uh, k, dk, f, dΩ, dΓ, nforward, nadjoint, Vρ, U, U0, V, bc, bc0, a, l, ctxs)
end

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

@testset "Functionals" begin

  @testset "Compliance, self-adjoint ($(design))" for design in (:P0, :P1)
    p = state_problem(; design)
    (; z, θ, S, f, dk, dΩ) = p
    J = Functional((u, θ) -> ∫(f * u)dΩ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), S, S, θ; scale = 1.0)
    g = functional_gradient!(zeros(length(z)), J, nothing)
    rates = taylor_rates(() -> functional_value(J, nothing), g, z, direction(length(z)))
    @test all(r -> 1.9 < r < 2.1, rates)
  end

  @testset "Boundary value with its own adjoint ($(design))" for design in (:P0, :P1)
    p = state_problem(; design)
    (; z, θ, S, A, dk, dΩ, dΓ) = p
    J = Functional((u, θ) -> ∫(u)dΓ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), S, A, θ; scale = 1.0)
    g = functional_gradient!(zeros(length(z)), J, nothing)
    rates = taylor_rates(() -> functional_value(J, nothing), g, z, direction(length(z)))
    @test all(r -> 1.9 < r < 2.1, rates)
  end

  @testset "Design-only functional (volume of θ)" begin
    p = state_problem()
    (; z, θ, dΩ) = p
    Vol = Functional(θ -> ∫(θ.ρ)dΩ, θ -> (δ -> ∫(δ)dΩ), θ; scale = 0.5)
    @test functional_value(Vol, nothing) ≈ sum(∫(θ.ρ)dΩ) / 0.5
    g = functional_gradient!(zeros(length(z)), Vol, nothing)
    rates = taylor_rates(() -> functional_value(Vol, nothing), g, z, direction(length(z)))
    @test all(r -> 1.9 < r < 2.1, rates)
  end

  @testset "Scale" begin
    p = state_problem()
    (; z, θ, S, f, dk, dΩ) = p
    Jf, DJf = (u, θ) -> ∫(f * u)dΩ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ)
    J1 = Functional(Jf, DJf, S, S, θ; scale = 1.0)
    Ji = Functional(Jf, DJf, S, S, θ; scale = :initial)
    J0 = functional_value(J1, nothing)
    @test functional_value(Ji, nothing) ≈ 1.0
    g1 = functional_gradient!(zeros(length(z)), J1, nothing)
    gi = functional_gradient!(zeros(length(z)), Ji, nothing)
    @test gi ≈ g1 ./ J0
    w = copy(get_values(z)); w .= clamp.(w .+ 0.05, 0, 1)
    set_values!(z, w)
    @test functional_value(Ji, nothing) ≈ functional_value(J1, nothing) / J0   # the scale stays the first value
    # gradient first: the scale is set from the value at that design
    Jg = Functional(Jf, DJf, S, S, θ; scale = :initial)
    gg = functional_gradient!(zeros(length(z)), Jg, nothing)
    @test functional_value(Jg, nothing) ≈ 1.0
    @test gg ≈ functional_gradient!(zeros(length(z)), J1, nothing) ./ functional_value(J1, nothing)
    for bad in (0.0, -1.0, NaN, Inf, :first)
      @test_throws ArgumentError Functional(Jf, DJf, S, S, θ; scale = bad)
    end
    @test_throws UndefKeywordError Functional(Jf, DJf, S, S, θ)
  end

  @testset "States are solved only when something changed" begin
    p = state_problem()
    (; z, θ, H, S, A, f, dk, dΩ, dΓ, nforward, nadjoint) = p
    C = Functional((u, θ) -> ∫(f * u)dΩ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), S, S, θ; scale = 1.0)
    B = Functional((u, θ) -> ∫(u)dΓ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), S, A, θ; scale = 1.0)
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
    J = Functional((u, θ) -> ∫(f * u)dΩ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), Sf, Sf, θ; scale = 1.0)
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
    J = Functional(θ -> ∫(θ.ρ * θ.r2)dΩ, θ -> (ρ = δ -> ∫(θ.r2 * δ)dΩ, r2 = δ -> ∫(θ.ρ * δ)dΩ), θ2; scale = 1.0)
    g = functional_gradient!(zeros(length(z)), J, nothing)
    rates = taylor_rates(() -> functional_value(J, nothing), g, z, direction(length(z)))
    @test all(r -> 1.9 < r < 2.1, rates)
    Jbad = Functional(θ -> ∫(θ.ρ)dΩ, θ -> (δ -> ∫(δ)dΩ), θ2; scale = 1.0)
    @test_throws ArgumentError functional_gradient!(zeros(length(z)), Jbad, nothing)
  end

  @testset "StateOperator: contract" begin
    p = state_problem()
    m = HyperShape.Operators.get_model(p.S)
    @test_throws ArgumentError StateOperator(m; depends = (1.0,), solve = (m, ctx) -> solve!(m))
    @test_throws UndefKeywordError StateOperator(m; solve = (m, ctx) -> solve!(m))
    @test_throws UndefKeywordError StateOperator(m; depends = (p.θ,))
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
    J = Functional((u, θ) -> ∫(u)dΓ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), S, A, θ; scale = 1.0)
    g = functional_gradient!(zeros(length(z)), J, nothing)
    @test all(r -> 1.9 < r < 2.1, taylor_rates(() -> functional_value(J, nothing), g, z, direction(length(z))))
    # adjoint built on the space with the Dirichlet values of the state: detected
    q = state_problem(; gD = 1.0, adjoint_space = :state)
    Jq = Functional((u, θ) -> ∫(u)dΓ, (u, w, θ) -> (δ -> ∫(-(q.dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)q.dΩ), q.S, q.A, q.θ; scale = 1.0)
    @test_throws ArgumentError functional_gradient!(zeros(length(q.z)), Jq, nothing)
    # self-adjoint shortcut with non-zero Dirichlet values: detected
    Js = Functional((u, θ) -> ∫(p.f * u)dΩ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), S, S, θ; scale = 1.0)
    @test_throws ArgumentError functional_gradient!(zeros(length(z)), Js, nothing)
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
    @test_throws ArgumentError Functional(Jf, DJf, S0, p.A, θ; scale = 1.0)
    @test_throws ArgumentError Functional(Jf, DJf, p.S, A1, θ; scale = 1.0)
  end

  @testset "Value kept until the design changes; ctx reaches every solve" begin
    p = state_problem()
    (; z, θ, S, A, dΓ, dk, dΩ, ctxs) = p
    nJ = Ref(0)
    J = Functional((u, θ) -> (nJ[] += 1; ∫(u)dΓ), (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), S, A, θ; scale = 1.0)
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
    J = Functional((u, θ) -> ∫(u)dΓ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), S, A, θ; scale = 1.0)
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
      S = StateOperator(StaticNonlinearModel(res, jac, U, V, bc; nls); depends = θ,
                        solve = (m, ctx) -> solve!(m; stepping = stepping[], RestartState = false))
      uh = get_state(S)
      A = StateOperator(StaticLinearModel(w -> ∫(w)dΓ, (p, w) -> jac(1.0)(uh, w, p), U0, V, bc0; ls = LUSolver());
                        depends = (θ, S), solve = (m, ctx) -> solve!(m; Assembly = true))
      Functional((u, θ) -> ∫(u)dΓ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * (1 + c * u * u) * ∇(u) ⋅ ∇(w) * δ)dΩ),
                 S, A, θ; scale = 1.0), S
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

  @testset "Lazy and compiled backends after changing z and β" begin
    pl = state_problem(; backend = :lazy)
    pc = state_problem(; backend = :compiled)
    funcs = map((pl, pc)) do p
      dΓ, dΩ, dk = p.dΓ, p.dΩ, p.dk
      Functional((u, θ) -> ∫(u)dΓ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), p.S, p.A, p.θ; scale = 1.0)
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

  @testset "Exported names do not clash with Gridap and HyperFEM" begin
    M = Module()
    Core.eval(M, :(using Gridap, HyperFEM, HyperShape))
    for name in (:StateOperator, :Functional, :functional_value, :functional_gradient!, :is_converged)
      @test Core.eval(M, name) === getfield(HyperShape, name)
    end
  end

end
