using Gridap
using Gridap.FESpaces
using HyperFEM
using HyperFEMKernels
using HyperShape
using HyperShape.Variables: update_fields!, set_values!, get_values, changed!, set_beta!, state_version
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
    (; z, nadjoint) = p
    J = p.compliance(scale = 1.0)
    g = functional_gradient!(zeros(length(z)), J, nothing)
    rates = taylor_rates(() -> functional_value(J, nothing), g, z, direction(length(z)))
    @test all(r -> 1.9 < r < 2.1, rates)
    @test nadjoint[] == 0                                   # self-adjoint: no adjoint solve
  end

  @testset "Boundary value with its own adjoint ($(design))" for design in (:P0, :P1)
    p = state_problem(; design)
    (; z) = p
    J = p.boundary(scale = 1.0)
    g = functional_gradient!(zeros(length(z)), J, nothing)
    rates = taylor_rates(() -> functional_value(J, nothing), g, z, direction(length(z)))
    @test all(r -> 1.9 < r < 2.1, rates)
  end

  @testset "dJdθ: a functional that reads θ directly ($(design))" for design in (:P0, :P1)
    p = state_problem(; design)
    (; z, θ, S, dΓ, dΩ) = p
    J = Functional((u, θ) -> ∫(u)dΓ + ∫(θ.ρ * θ.ρ)dΩ, S; dJdu = u -> (w -> ∫(w)dΓ),
                   dJdθ = (u, θ) -> (δ -> ∫(2 * θ.ρ * δ)dΩ), scale = 1.0)
    g = functional_gradient!(zeros(length(z)), J, nothing)
    @test all(r -> 1.9 < r < 2.1, taylor_rates(() -> functional_value(J, nothing), g, z, direction(length(z))))
    # with the same adjoint, the gradient is the one of the boundary value plus the design part
    Vq = Functional(θ -> ∫(θ.ρ * θ.ρ)dΩ, θ; dJdθ = θ -> (δ -> ∫(2 * θ.ρ * δ)dΩ), scale = 1.0)
    gb = functional_gradient!(zeros(length(z)), p.boundary(scale = 1.0), nothing)
    gq = functional_gradient!(zeros(length(z)), Vq, nothing)
    @test g ≈ gb .+ gq rtol = 1e-12
  end

  @testset "SymmetricState: the adjoint with the factorisation of the state ($(design))" for design in (:P0, :P1)
    p = state_problem(; design)
    (; z, θ, dΓ, dΩ, dk, a, l, U, V, bc, sens) = p
    m = StaticLinearModel(l, a, U, V, bc; ls = LUSolver())
    Ss = StateOperator(m, θ; adjoint = AdjointProblem(SymmetricState(); sensitivity = sens))
    J = Functional((u, θ) -> ∫(u)dΓ, Ss; dJdu = u -> (w -> ∫(w)dΓ), scale = 1.0)
    g = functional_gradient!(zeros(length(z)), J, nothing)
    @test all(r -> 1.9 < r < 2.1, taylor_rates(() -> functional_value(J, nothing), g, z, direction(length(z))))
    @test g ≈ functional_gradient!(zeros(length(z)), p.boundary(scale = 1.0), nothing) rtol = 1e-10
    @test HyperShape.Operators.num_solves(Ss.adjoint) >= 1
    # a state that is not a StaticLinearModel with one right-hand side cannot use it
    multi = StaticLinearModel([v -> ∫(v)p.dΩ, v -> ∫(2v)p.dΩ], a, p.U0, V, p.bc0; ls = LUSolver())
    @test_throws ArgumentError StateOperator(multi, θ; solve = (m, ctx) -> solve!(m),
                                             adjoint = AdjointProblem(SymmetricState(); sensitivity = sens))
    @test_throws ArgumentError AdjointProblem(SymmetricState(); sensitivity = sens, solve = (m, b, c, s) -> b)
    # get_adjoint: the adjoint field of the last gradient
    @test get_free_dof_values(get_adjoint(Ss)) == get_free_dof_values(Ss.adjoint.p)
  end

  @testset "AdjointProblem without an operator: only self-adjoint functionals" begin
    p = state_problem()
    (; z, θ, f, dΩ, dΓ, a, l, U, V, bc, sens, nadjoint) = p
    S0 = StateOperator(StaticLinearModel(l, a, U, V, bc; ls = LUSolver()), θ; adjoint = AdjointProblem(sensitivity = sens))
    C0 = Functional((u, θ) -> ∫(f * u)dΩ, S0; dJdu = SelfAdjoint(), scale = 1.0)
    g0 = functional_gradient!(zeros(length(z)), C0, nothing)
    @test g0 ≈ functional_gradient!(zeros(length(z)), p.compliance(scale = 1.0), nothing) rtol = 1e-12
    @test get_free_dof_values(get_adjoint(S0)) == get_free_dof_values(get_state(S0))    # p = u
    err = try Functional((u, θ) -> ∫(u)dΓ, S0; dJdu = u -> (w -> ∫(w)dΓ), scale = 1.0); nothing catch e; e end
    @test err isa ArgumentError && occursin("no operator", err.msg)
    @test_throws ArgumentError AdjointProblem(nothing; sensitivity = sens, depends = (θ,))
    @test_throws ArgumentError get_adjoint(StateOperator(get_model(p.S), θ))
    # design-only functionals: dJdθ (the old keyword explains the new one)
    err = try Functional(θ -> ∫(θ.ρ)dΩ, θ; derivative = θ -> (δ -> ∫(δ)dΩ), scale = 1.0); nothing catch e; e end
    @test err isa ArgumentError && occursin("dJdθ", err.msg)
    @test_throws ArgumentError Functional(θ -> ∫(θ.ρ)dΩ, θ; scale = 1.0)
  end

  @testset "Design-only functional (volume of θ)" begin
    p = state_problem()
    (; z, θ, dΩ) = p
    Vol = Functional(θ -> ∫(θ.ρ)dΩ, θ; dJdθ = θ -> (δ -> ∫(δ)dΩ), scale = 0.5)
    @test functional_value(Vol, nothing) ≈ sum(∫(θ.ρ)dΩ) / 0.5
    g = functional_gradient!(zeros(length(z)), Vol, nothing)
    rates = taylor_rates(() -> functional_value(Vol, nothing), g, z, direction(length(z)))
    @test all(r -> 1.9 < r < 2.1, rates)
  end

  @testset "Scale" begin
    p = state_problem()
    (; z) = p
    J1 = p.compliance(scale = 1.0)
    Ji = p.compliance(scale = :initial)
    J0 = functional_value(J1, nothing)
    @test functional_value(Ji, nothing) ≈ 1.0
    g1 = functional_gradient!(zeros(length(z)), J1, nothing)
    gi = functional_gradient!(zeros(length(z)), Ji, nothing)
    @test gi ≈ g1 ./ J0
    w = copy(get_values(z)); w .= clamp.(w .+ 0.05, 0, 1)
    set_values!(z, w)
    @test functional_value(Ji, nothing) ≈ functional_value(J1, nothing) / J0   # the scale stays the first value
    # gradient first: the scale is set from the value at that design
    Jg = p.compliance(scale = :initial)
    gg = functional_gradient!(zeros(length(z)), Jg, nothing)
    @test functional_value(Jg, nothing) ≈ 1.0
    @test gg ≈ functional_gradient!(zeros(length(z)), J1, nothing) ./ functional_value(J1, nothing)
    for bad in (0.0, -1.0, NaN, Inf, :first)
      @test_throws ArgumentError p.compliance(scale = bad)
    end
    @test_throws UndefKeywordError p.compliance()
  end

  @testset "States are solved only when something changed; one assembly of the adjoint per design" begin
    p = state_problem()
    (; z, θ, H, S, nforward, nadjoint, nassembly) = p
    C = p.compliance(scale = 1.0)
    B = p.boundary(scale = 1.0)
    g = zeros(length(z))
    functional_value(C, nothing); functional_value(B, nothing)
    functional_gradient!(g, C, nothing); functional_gradient!(g, B, nothing)
    @test (nforward[], nadjoint[], nassembly[]) == (1, 1, 1)   # one forward solve shared, one adjoint
    vS = state_version(S)
    functional_gradient!(g, B, nothing)                       # same design: a back substitution, no assembly
    @test (nforward[], nadjoint[], nassembly[]) == (1, 2, 1) && state_version(S) == vS
    changed!(z)
    functional_value(C, nothing); functional_gradient!(g, B, nothing)
    @test (nforward[], nadjoint[], nassembly[]) == (2, 3, 2)
    set_beta!(H, 6.0)                                      # a new β changes θ: states solved again
    functional_value(C, nothing)
    @test nforward[] == 3
    # the state solved again with the same dependencies (invalidate!): the adjoint matrix is assembled again
    functional_gradient!(g, B, nothing)
    na = nassembly[]
    HyperShape.Operators.invalidate!(S)
    functional_gradient!(g, B, nothing)
    @test nassembly[] == na + 1
    # a zero weight seeds nothing: no adjoint solve
    n0 = nadjoint[]
    functional_gradient!(g, (C, B), [1.0, 0.0], nothing)
    @test nadjoint[] == n0
  end

  @testset "Seeds on a vector-valued space with the compiled backend: no fallback" begin
    model = CartesianDiscreteModel((0, 2, 0, 1), (8, 4))
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "left", [1, 3, 7]); add_tag_from_tags!(labels, "right", [2, 4, 8])
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)
    dΓ = Measure(BoundaryTriangulation(model; tags = "right"), 2)
    Vρ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
    z, x, θ = single_field_design(:ρ, Vρ; bounds = (0.0, 1.0), init = [0.3 + 0.5 * abs(sin(2.3i)) for i in 1:num_free_dofs(Vρ)],
                                  map = Identity())
    E(ρ) = 1e-3 + ρ^3; dE(ρ) = 3ρ^2
    σ(ε) = 0.5 * tr(ε) * one(ε) + ε
    t = VectorValue(0.0, -1.0)
    bc = BoundaryConditions(BoundaryCondition("left", VectorValue(0.0, 0.0)))
    V = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1), bc; conformity = :H1)
    U = TrialFESpace(V, bc, 1.0)
    a(u, v) = ∫((E ∘ θ.ρ) * (ε(v) ⊙ (σ ∘ ε(u))))dΩ
    m = StaticLinearModel(v -> ∫(t ⋅ v)dΓ, a, U, V, bc; ls = LUSolver())
    uh = get_state(m)
    S = StateOperator(m, θ; adjoint = AdjointProblem(SymmetricState();
                                                     sensitivity = (u, q) -> (δ -> ∫((dE ∘ θ.ρ) * (ε(q) ⊙ (σ ∘ ε(u))) * δ)dΩ)))
    g = VectorValue(1.0, 0.5)
    J = Functional((u, θ) -> ∫(g ⋅ u)dΓ, S; dJdu = u -> (w -> ∫(g ⋅ w)dΓ), scale = 1.0)
    gz = @test_logs functional_gradient!(zeros(length(z)), J, nothing)     # no warning of HyperFEMKernels
    @test all(r -> 1.9 < r < 2.1, taylor_rates(() -> functional_value(J, nothing), gz, z, direction(length(z))))
  end

  @testset "A tuple of one functional is the functional" begin
    for scale in (3.7, :initial)
      p = state_problem()
      z = p.z
      B = p.boundary(; scale)
      g1 = functional_gradient!(zeros(length(z)), B, nothing)
      g2 = functional_gradient!(zeros(length(z)), (B,), [1.0], nothing)
      @test g1 == g2                                       # bit for bit: the scale rule divides by the same s
      @test functional_gradient!(zeros(length(z)), (B,), [2.5], nothing) ≈ 2.5 .* g1 rtol = 1e-14
    end
  end

  @testset "Constraint: bounds, :bound scale and errors" begin
    p = state_problem()
    (; θ, dΩ) = p
    mk(scale) = Functional(θ -> ∫(θ.ρ)dΩ, θ; dJdθ = θ -> (δ -> ∫(δ)dΩ), scale)
    c = Constraint(mk(1.0))
    @test (c.lo, c.hi) == (-Inf, 0.0)                     # a functional alone: F ≤ 0
    @test Constraint(c) === c
    c = Constraint(mk(1.0); min = 0.2)
    @test (c.lo, c.hi) == (0.2, Inf) && HyperShape.Functionals._reference(c) == 0.2
    c = Constraint(mk(1.0); min = -1, max = 3)
    @test (c.lo, c.hi) == (-1.0, 3.0) && HyperShape.Functionals._reference(c) == 3.0
    c = Constraint(mk(1.0); equal = 0.5)
    @test c.lo == c.hi == 0.5
    cb = Constraint(mk(:bound); min = -4.0, max = 2.0)
    @test HyperShape.Functionals.get_scale(cb.F) == 4.0   # largest finite bound in absolute value
    @test_throws ArgumentError Constraint(mk(:bound); max = 0.0)
    @test_throws ArgumentError Constraint(mk(1.0); max = 1.0, min = 2.0)
    @test_throws ArgumentError Constraint(mk(1.0); equal = 1.0, max = 2.0)
    @test_throws ArgumentError Constraint(mk(1.0); max = NaN)
    @test_throws ArgumentError Constraint(mk(1.0); min = Inf)
    @test_throws ArgumentError Constraint(mk(1.0); max = Inf)                # no finite bound
    @test_throws ArgumentError Constraint(mk(1.0); min = -Inf)
    @test_throws ArgumentError Constraint(mk(:bound); max = Inf)
    Fb = mk(:bound)
    Constraint(Fb; max = 2.0)
    @test HyperShape.Functionals.get_scale(Constraint(Fb; max = -2.0).F) == 2.0   # the same scale: accepted
    @test_throws ArgumentError Constraint(Fb; max = 10.0)                    # another scale for the same functional
    @test_throws ArgumentError functional_value(mk(:bound), nothing)   # :bound outside a Constraint
  end

  @testset "Several functionals in one reverse sweep" begin
    for vb in (:compiled, :lazy)
      p = state_problem(; vector_backend = vb)
      (; z, θ, dΩ, dΓ, nadjoint, Sf) = p
      C = p.compliance(scale = 2.0)
      B = p.boundary(scale = :initial)
      B2 = Functional((u, θ) -> ∫(u * u)dΓ, p.S; dJdu = u -> (w -> ∫(2 * u * w)dΓ), scale = 1.0)
      Vol = Functional(θ -> ∫(θ.ρ)dΩ, θ; dJdθ = θ -> (δ -> ∫(δ)dΩ), scale = 0.5)
      Fs, w = (C, B, B2, Vol), [1.0, -0.7, 0.4, 2.5]
      gs = [functional_gradient!(zeros(length(z)), F, nothing) for F in Fs]
      n0, f0 = nadjoint[], HyperShape.Operators.num_solves(Sf)
      g = functional_gradient!(zeros(length(z)), Fs, w, nothing)
      @test g ≈ sum(w[i] .* gs[i] for i in eachindex(Fs)) rtol = 1e-10
      @test nadjoint[] == n0 + 1                           # one adjoint solve for B and B2 (C is self-adjoint)
      @test HyperShape.Operators.num_solves(Sf) == f0 + 1  # one transposed filter solve for the four
      # Taylor test of the combination
      val() = sum(w[i] * functional_value(Fs[i], nothing) for i in eachindex(Fs))
      @test all(r -> 1.9 < r < 2.1, taylor_rates(val, g, z, direction(length(z))))
      @test_throws DimensionMismatch functional_gradient!(zeros(length(z)), Fs, [1.0], nothing)
      θ2 = PhysicalFields(p.x; ρ = Identity())
      V2 = Functional(θ -> ∫(θ.ρ)dΩ, θ2; dJdθ = θ -> (δ -> ∫(δ)dΩ), scale = 1.0)
      # functionals on different physical fields of the same z: one sweep per group, added
      gC, gV2 = functional_gradient!(zeros(length(z)), C, nothing), functional_gradient!(zeros(length(z)), V2, nothing)
      @test functional_gradient!(zeros(length(z)), (C, V2, Vol), [1.0, -2.0, 0.5], nothing) ≈ gC .- 2 .* gV2 .+ 0.5 .* gs[4] rtol = 1e-10
    end
  end

  @testset "Failed forward solve" begin
    p = state_problem()
    (; z, θ, f, dΩ) = p
    converge = Ref(false)
    Sf = StateOperator(get_model(p.S); depends = (θ,), adjoint = p.adj,
                       solve = (m, ctx) -> (solve!(m; Assembly = true), converge[]))
    J = Functional((u, θ) -> ∫(f * u)dΩ, Sf; dJdu = SelfAdjoint(), fields = θ, scale = 1.0)
    @test isnan(functional_value(J, nothing))
    g = functional_gradient!(zeros(length(z)), J, nothing)
    @test all(isnan, g)
    converge[] = true
    @test isnan(functional_value(J, nothing))              # the failed solve is not repeated for the same design
    changed!(z)
    @test isfinite(functional_value(J, nothing))
    @test all(isfinite, functional_gradient!(g, J, nothing))
  end

  @testset "Several physical fields: derivative as a named tuple" begin
    p = state_problem()
    (; z, x, f, dk, dΩ) = p
    θ2 = PhysicalFields(x; ρ = Identity(), r2 = :ρ => TanhProjection(β = 2.0, η = 0.5))
    J = Functional(θ -> ∫(θ.ρ * θ.r2)dΩ, θ2; dJdθ = θ -> (ρ = δ -> ∫(θ.r2 * δ)dΩ, r2 = δ -> ∫(θ.ρ * δ)dΩ), scale = 1.0)
    g = functional_gradient!(zeros(length(z)), J, nothing)
    rates = taylor_rates(() -> functional_value(J, nothing), g, z, direction(length(z)))
    @test all(r -> 1.9 < r < 2.1, rates)
    Jbad = Functional(θ -> ∫(θ.ρ)dΩ, θ2; dJdθ = θ -> (δ -> ∫(δ)dΩ), scale = 1.0)
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
    (; z, θ, S, f, dΩ) = p
    J = p.boundary(scale = 1.0)
    g = functional_gradient!(zeros(length(z)), J, nothing)
    @test all(r -> 1.9 < r < 2.1, taylor_rates(() -> functional_value(J, nothing), g, z, direction(length(z))))
    # adjoint built on the space with the Dirichlet values of the state: detected when the state is built
    @test_throws ArgumentError state_problem(; gD = 1.0, adjoint_space = :state)
    # self-adjoint shortcut with non-zero Dirichlet values: detected when the functional is built
    err = try
      Functional((u, θ) -> ∫(f * u)dΩ, S; dJdu = SelfAdjoint(), scale = 1.0)
      nothing
    catch e
      e
    end
    @test err isa ArgumentError && occursin("SelfAdjoint", err.msg)
  end

  @testset "Adjoint of a failed state is not solved" begin
    p = state_problem()
    (; z, θ, nadjoint) = p
    converge = Ref(false)
    Sf = StateOperator(get_model(p.S); depends = θ, adjoint = p.adj,
                       solve = (m, ctx) -> (solve!(m; Assembly = true), converge[]))
    J = Functional((u, θ) -> ∫(u)p.dΓ, Sf; dJdu = u -> (w -> ∫(w)p.dΓ), scale = 1.0)
    @test all(isnan, functional_gradient!(zeros(length(z)), J, nothing)) && nadjoint[] == 0
    converge[] = true
    changed!(z)
    @test all(isfinite, functional_gradient!(zeros(length(z)), J, nothing)) && nadjoint[] == 1
  end

  @testset "Functional with a state: what is checked when it is built" begin
    p = state_problem()
    (; θ, dΓ, dΩ, dk) = p
    m = get_model(p.S)
    Jf = (u, θ) -> ∫(u)dΓ
    dJdu = u -> (w -> ∫(w)dΓ)
    # a state without an adjoint only gives values
    S0 = StateOperator(m, θ)
    err = try Functional(Jf, S0; dJdu, scale = 1.0); nothing catch e; e end
    @test err isa ArgumentError && occursin("AdjointProblem", err.msg)
    # dJdu is mandatory; the old keywords say what replaced them
    @test_throws ArgumentError Functional(Jf, p.S; scale = 1.0)
    err = try Functional(Jf, p.S; derivative = (u, w, θ) -> nothing, adjoint = :self, scale = 1.0); nothing catch e; e end
    @test err isa ArgumentError && occursin("dJdu", err.msg) && occursin("AdjointProblem", err.msg)
    # a state that depends on another state: not supported yet (the upstream sensitivity would be missing)
    Sc = StateOperator(m, (θ, p.S); adjoint = p.adj)
    err = try Functional(Jf, Sc; dJdu, scale = 1.0); nothing catch e; e end
    @test err isa ArgumentError && occursin("another state", err.msg)
    # the adjoint must be an AdjointProblem
    @test_throws ArgumentError StateOperator(m, θ; adjoint = :self)
    @test_throws ArgumentError AdjointProblem(1.0; sensitivity = p.sens)
    @test_throws ArgumentError AdjointProblem(p.model_p; sensitivity = p.sens, backend = :gpu)
  end

  @testset "Value kept until the design changes; ctx reaches every solve" begin
    p = state_problem()
    (; z, θ, S, dΓ, ctxs) = p
    nJ = Ref(0)
    J = Functional((u, θ) -> (nJ[] += 1; ∫(u)dΓ), S; dJdu = u -> (w -> ∫(w)dΓ), fields = θ, scale = 1.0)
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
    apply!(S, nothing)
    @test (@allocated apply!(S, nothing)) == 0
  end

  @testset "Taylor test after a change of β" begin
    p = state_problem()
    (; z, H, nassembly) = p
    J = p.boundary(scale = 1.0)
    functional_gradient!(zeros(length(z)), J, nothing)
    n = nassembly[]
    set_beta!(H, 8.0)
    g = functional_gradient!(zeros(length(z)), J, nothing)
    @test nassembly[] == n + 1                              # the adjoint matrix is assembled again
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
    z, x, θ = single_field_design(:ρ, Vρ; bounds = (0.0, 1.0), init = ρ0, map = Identity())
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
      model_u = StaticNonlinearModel(res, jac, U, V, bc; nls)
      uh = get_state(model_u)
      adj = AdjointProblem(StaticLinearModel((p, w) -> jac(1.0)(uh, w, p), U0, V, bc0; ls = LUSolver());
                           sensitivity = (u, q) -> (δ -> ∫((dk ∘ θ.ρ) * (1 + c * u * u) * ∇(u) ⋅ ∇(q) * δ)dΩ))
      S = StateOperator(model_u, θ; adjoint = adj, solve = (m, ctx) -> solve!(m; stepping = stepping[], RestartState = false))
      @test S.adjoint.op.depends === (θ,)
      @test_throws ArgumentError Functional((u, θ) -> ∫(u)dΓ, S; dJdu = SelfAdjoint(), scale = 1.0)   # p = u is wrong
      Functional((u, θ) -> ∫(u)dΓ, S; dJdu = u -> (w -> ∫(w)dΓ), scale = 1.0), S
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

  @testset "Default solve, dependencies, AdjointProblem and the keyword form of Functional" begin
    p = state_problem()
    (; z, θ, S, f, dk, dΩ, dΓ, a, l, U, U0, V, bc, bc0, sens) = p
    # default solves of a StaticLinearModel: the same states and gradients as the explicit solves
    Sd = StateOperator(StaticLinearModel(l, a, U, V, bc; ls = LUSolver()), θ;
                       adjoint = AdjointProblem(StaticLinearModel((p, w) -> a(w, p), U0, V, bc0; ls = LUSolver());
                                                sensitivity = sens))
    @test Sd.solve === HyperShape.Operators.default_solve(StateOperator, Sd.model)
    @test Sd.adjoint.op.solve === HyperShape.Operators.default_solve(LinearOperator, Sd.adjoint.op.model)
    @test Sd.adjoint.op.depends === (θ,)
    Jd = Functional((u, θ) -> ∫(u)dΓ, Sd; dJdu = u -> (w -> ∫(w)dΓ), scale = 1.0)
    Je = p.boundary(scale = 1.0)
    @test Jd.θ === θ                                   # θ deduced from the dependencies of S
    gd = functional_gradient!(zeros(length(z)), Jd, nothing)   # before anything solves p.S
    @test functional_value(Jd, nothing) == functional_value(Je, nothing)
    @test gd ≈ functional_gradient!(zeros(length(z)), Je, nothing) rtol = 1e-12
    @test get_free_dof_values(get_state(Sd)) == get_free_dof_values(get_state(S))
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
    # AdjointProblem: extra dependencies without repetitions, homogeneous trial space
    Se = StateOperator(Sd.model, θ; adjoint = AdjointProblem(Sd.adjoint.op.model; sensitivity = sens, depends = (θ, z)))
    @test Se.adjoint.op.depends === (θ, z)
    bc1 = BoundaryConditions(BoundaryCondition("left", 1.0))
    U1 = TrialFESpace(V, bc1, 1.0)
    @test_throws ArgumentError StateOperator(Sd.model, θ;
      adjoint = AdjointProblem(StaticLinearModel((p, w) -> a(w, p), U1, V, bc1; ls = LUSolver()); sensitivity = sens))
    # Functional: fields deduced or given
    θ2 = PhysicalFields(p.x; ρ = Identity())
    S2 = StateOperator(Sd.model, (θ, θ2); adjoint = p.adj)
    @test_throws ArgumentError Functional((u, θ) -> ∫(u)dΓ, S2; dJdu = SelfAdjoint(), scale = 1.0)
    err = try Functional((u, θ) -> ∫(u)dΓ, S2; dJdu = SelfAdjoint(), scale = 1.0, fields = θ); nothing catch e; e end
    @test err isa ArgumentError && occursin("several PhysicalFields", err.msg)   # the sensitivity through θ2 would be missing
    @test_throws ArgumentError Functional((u, θ) -> ∫(u)dΓ, StateOperator(Sd.model, z; adjoint = p.adj);
                                          dJdu = SelfAdjoint(), scale = 1.0)
    @test_throws ArgumentError Functional((u, θ) -> ∫(u)dΓ, Sd; dJdu = SelfAdjoint(), scale = 1.0,
                                          fields = θ2)                              # Sd does not depend on θ2
    # repeated dependencies are kept once
    @test StateOperator(Sd.model, (θ, θ)).depends === (θ,)
    @test Functional((u, θ) -> ∫(u)dΓ, StateOperator(Sd.model, (θ, θ); adjoint = p.adj); dJdu = SelfAdjoint(),
                     scale = 1.0).θ === θ
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
    @test HyperShape.Operators.dirichlet_values(_NoSpaces()) === nothing
  end

  @testset "Lazy and compiled backends after changing z and β" begin
    pl = state_problem(; backend = :lazy)
    pc = state_problem(; backend = :compiled)
    funcs = map(p -> p.boundary(scale = 1.0), (pl, pc))
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
      (; z, θ, dΩ) = p
      C = p.compliance(scale = 1.0)
      B = p.boundary(scale = 1.0)
      Vol = Functional(θ -> ∫(θ.ρ)dΩ, θ; dJdθ = θ -> (δ -> ∫(δ)dΩ), scale = 1.0)
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

  @testset "Exported names do not clash with Gridap, HyperFEM and LinearAlgebra" begin
    M = Module()
    Core.eval(M, :(using LinearAlgebra, Gridap, HyperFEM, HyperShape))
    for name in names(HyperShape)
      (name === :HyperShape || !Base.isexported(HyperShape, name)) && continue
      @test Core.eval(M, name) === getfield(HyperShape, name)
    end
  end

end
