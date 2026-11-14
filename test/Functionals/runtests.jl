using Gridap
using Gridap.FESpaces
using HyperFEM
using HyperFEMKernels
using HyperShape
using HyperShape.Variables: update_fields!, set_values!, get_values, changed!, set_beta!, version
using HyperShape.Operators: apply!
using LinearAlgebra
using Test

# Heat-conduction-like problem: -∇⋅(k(θ.ρ) ∇u) = f in Ω, u = 0 on the left side, with k(ρ) = kmin + ρ³.
function state_problem(; n = 8, design = :P0, backend = :lazy)
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
  bc = BoundaryConditions(BoundaryCondition("left", 0.0))
  V = TestFESpace(Ω, ReferenceFE(lagrangian, Float64, 1), bc, conformity = :H1)
  U = TrialFESpace(V, bc, 1.0)
  a(u, v) = ∫((k ∘ θ.ρ) * ∇(u) ⋅ ∇(v))dΩ
  l(v) = ∫(f * v)dΩ
  nforward, nadjoint = Ref(0), Ref(0)
  model_u = StaticLinearModel(l, a, U, V, bc; ls = LUSolver(), backend)
  S = StateOperator(model_u; depends = (θ,), solve = (m, ctx) -> (nforward[] += 1; solve!(m; Assembly = true)))
  uh = get_state(S)
  # adjoint of J = ∫_Γ u dΓ: a(w, p) = ∫_Γ w dΓ (a is symmetric)
  model_p = StaticLinearModel(w -> ∫(w)dΓ, (p, w) -> a(w, p), U, V, bc; ls = LUSolver(), backend)
  A = StateOperator(model_p; depends = (θ, S), solve = (m, ctx) -> (nadjoint[] += 1; solve!(m; Assembly = true)))
  (; z, x, θ, H, S, A, uh, k, dk, f, dΩ, dΓ, nforward, nadjoint, Vρ)
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
    Sf = StateOperator(HyperShape.Operators.get_model(p.S); depends = (θ,),
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
    @test_throws ArgumentError StateOperator(m; depends = p.θ, solve = (m, ctx) -> solve!(m))
    @test_throws ArgumentError StateOperator(m; depends = (1.0,), solve = (m, ctx) -> solve!(m))
    @test_throws UndefKeywordError StateOperator(m; solve = (m, ctx) -> solve!(m))
    @test_throws UndefKeywordError StateOperator(m; depends = (p.θ,))
    @test get_state(p.S) === get_state(p.S)                # fixed object for the adjoint forms
    ctxs = Any[]
    S2 = StateOperator(m; depends = (p.θ,), solve = (m, ctx) -> (push!(ctxs, ctx); solve!(m; Assembly = true)))
    @test apply!(S2, (iteration = 3,)) && ctxs == [(iteration = 3,)]
  end

  @testset "Same gradient with the lazy and the compiled backends" begin
    pl = state_problem(; backend = :lazy)
    pc = state_problem(; backend = :compiled)
    grads = map((pl, pc)) do p
      dΓ, dΩ, dk = p.dΓ, p.dΩ, p.dk
      J = Functional((u, θ) -> ∫(u)dΓ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), p.S, p.A, p.θ; scale = 1.0)
      functional_gradient!(zeros(length(p.z)), J, nothing)
    end
    @test maximum(abs, grads[1] .- grads[2]) <= 1e-10 * maximum(abs, grads[1])
  end

  @testset "Exported names do not clash with Gridap and HyperFEM" begin
    M = Module()
    Core.eval(M, :(using Gridap, HyperFEM, HyperShape))
    for name in (:StateOperator, :Functional, :functional_value, :functional_gradient!)
      @test Core.eval(M, name) === getfield(HyperShape, name)
    end
  end

end
