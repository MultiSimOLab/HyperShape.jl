using Gridap
using Gridap.FESpaces
using HyperFEM
using HyperShape
using HyperShape.Variables: get_values, set_values!, version, set_beta!
using HyperShape.Operators: apply!
using HyperShape.Checks: ConsistencyReport
using LinearAlgebra
using Test

# results of a report by label prefix
_results(r, prefix) = filter(x -> startswith(x.label, prefix), r.results)
_all_passed(r, prefix) = (xs = _results(r, prefix); !isempty(xs) && all(x -> x.passed, xs))

# design-only setup with a filter given by its bilinear form (to break symmetry or dependencies)
function filter_problem(af_of; n = 6, depends_of = x -> ())
  model = CartesianDiscreteModel((0, 1, 0, 1), (n, n))
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  V0 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
  Vf = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
  z = OptVariables(ρ = FieldVariable(V0; bounds = (0.0, 1.0), init = [0.3 + 0.4 * abs(sin(1.3i)) for i in 1:num_free_dofs(V0)]))
  x = DesignFields(z; ρ = Identity())
  af = af_of(x, dΩ)
  Sf = LinearOperator(StaticLinearModel(af, TrialFESpace(Vf), Vf, NothingBC(); ls = LUSolver()); depends = depends_of(x),
                      solve = (m, b, ctx, stale) -> solve!(m, b; Assembly = stale))
  θ = PhysicalFields(x; ρ = Heaviside(β = 2.0, η = 0.5) ∘ HelmholtzFilter(Sf, dΩ))
  F = Functional(θ -> ∫(θ.ρ * θ.ρ)dΩ, θ; derivative = θ -> (δ -> ∫(2 * θ.ρ * δ)dΩ), scale = 1.0)
  (; z, x, θ, F, dΩ)
end

symmetric_filter(x, dΩ) = (u, v) -> ∫(0.01 * ∇(u) ⋅ ∇(v) + u * v)dΩ

@testset "Checks" begin

  @testset "Consistent functionals pass ($(design))" for design in (:P0, :P1)
    p = state_problem(; design)
    (; z, θ, S, A, f, dk, dΩ, dΓ, a, l) = p
    C = Functional((u, θ) -> ∫(f * u)dΩ, S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = :self, fields = θ, scale = :initial)
    B = Functional((u, θ) -> ∫(u)dΓ, S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = A, fields = θ, scale = 2.0)
    Vol = Functional(θ -> ∫(θ.ρ)dΩ, θ; derivative = θ -> (δ -> ∫(δ)dΩ), scale = 1.0)
    z0 = copy(get_values(z))
    r = check_consistency((compliance = C, boundary = B, volume = Vol);
                          tangents = (S => ((u, v) -> a(u, v) - l(v), (u, du, v) -> a(du, v)),))
    @test r isa ConsistencyReport
    @test r.passed
    @test length(r.results) == 3 + 1 + 1 + 1 + 3   # gradients, chain, filter, tangent, dependencies
    @test _all_passed(r, "gradient of compliance") && _all_passed(r, "gradient of boundary")
    @test _all_passed(r, "design chain of θ.ρ") && _all_passed(r, "symmetry of filter 1")
    @test _all_passed(r, "tangent") && _all_passed(r, "dependencies")
    @test get_values(z) == z0                       # z restored
    @test get_free_dof_values(p.x.ρ) == z0          # and the fields and states with it
    ρvol = sum(∫(θ.ρ)dΩ)
    uref = copy(get_free_dof_values(p.uh))
    set_values!(z, z0); apply!(S, nothing)
    @test get_free_dof_values(p.uh) ≈ uref rtol = 1e-12
    @test HyperShape.Operators.is_converged(S)
    @test sum(∫(θ.ρ)dΩ) ≈ ρvol rtol = 1e-14
    @test isnan(C.scale[]) && B.scale[] == 2.0      # scales restored
    @test functional_value(C, nothing) ≈ 1.0        # :initial still taken at the first real evaluation
    s = sprint(show, MIME("text/plain"), r)
    @test occursin("passed", s) && occursin("gradient of compliance", s)
    @test occursin("of 9 passed", sprint(show, r))
  end

  @testset "A wrong derivative is found and located" begin
    p = state_problem()
    (; θ, S, f, dk, dΩ, a, l) = p
    # factor 2 in DJform
    C = Functional((u, θ) -> ∫(f * u)dΩ, S; derivative = (u, w, θ) -> (δ -> ∫(-2 * (dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = :self, fields = θ, scale = 1.0)
    r = check_consistency(C; tangents = S => ((u, v) -> a(u, v) - l(v), (u, du, v) -> 2 * a(du, v)))
    @test !r.passed
    @test !_all_passed(r, "gradient of functional 1")
    @test _all_passed(r, "design chain")            # the maps are fine: the error is in DJform
    @test _all_passed(r, "symmetry")
    @test !_all_passed(r, "tangent")                # wrong tangent
    @test _all_passed(r, "dependencies")
  end

  @testset "Non-symmetric filter" begin
    q = filter_problem((x, dΩ) -> (u, v) -> ∫(0.01 * ∇(u) ⋅ ∇(v) + u * v + 0.5 * (∇(u) ⋅ VectorValue(1.0, 0.0)) * v)dΩ)
    r = check_consistency(q.F)
    @test !_all_passed(r, "symmetry of filter 1")
    @test !_all_passed(r, "design chain")           # the transposed filter is wrong
    q = filter_problem(symmetric_filter)
    @test check_consistency(q.F).passed
  end

  @testset "Undeclared dependency of a filter matrix" begin
    # the matrix of the filter reads the design field x.ρ
    af(x, dΩ) = (u, v) -> ∫(0.01 * ∇(u) ⋅ ∇(v) + (1 + x.ρ) * u * v)dΩ
    q = filter_problem(af)                          # depends = (): the matrix is never assembled again
    r = check_consistency(q.F)
    @test !_all_passed(r, "dependencies")
    q = filter_problem(af; depends_of = x -> x)     # declared: no difference after solving again
    r = check_consistency(q.F)
    @test _all_passed(r, "dependencies")
    @test _all_passed(r, "symmetry")
  end

  @testset "Arguments" begin
    p = state_problem()
    q = filter_problem(symmetric_filter)
    dΩ = p.dΩ
    Vol = Functional(θ -> ∫(θ.ρ)dΩ, p.θ; derivative = θ -> (δ -> ∫(δ)dΩ), scale = 1.0)
    @test_throws ArgumentError check_consistency(Vol, q.F)                    # different z
    @test_throws ArgumentError check_consistency()
    @test_throws ArgumentError check_consistency(Vol; steps = 1)
    n = length(p.z)
    @test_throws ArgumentError check_consistency(Vol; direction = zeros(n))
    @test_throws DimensionMismatch check_consistency(Vol; direction = ones(n + 1))
    @test_throws ArgumentError check_consistency(Vol; direction = ones(n), h0 = 10.0)   # leaves the bounds
    # a given direction and step
    r = check_consistency(Vol; direction = [0.1 * sin(1.7i) for i in 1:n], h0 = 1e-2, steps = 3)
    @test r.passed
    @test length(_results(r, "gradient")) == 1
    # values at a bound get a zero direction
    v = copy(get_values(p.z)); v[1] = 0.0; v[2] = 1.0
    set_values!(p.z, v)
    r = check_consistency(Vol)
    @test r.passed && get_values(p.z) == v
  end

  @testset "Sensitivity, saturated designs and keywords" begin
    p = state_problem(; design = :P1)
    (; θ, S, f, dk, dΩ) = p
    DJ(c) = (u, w, θ) -> (δ -> ∫(-c * (dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ)
    C = Functional((u, θ) -> ∫(f * u)dΩ, S; derivative = DJ(1.0), adjoint = :self, fields = θ, scale = 1.0)
    @test _all_passed(check_consistency(C), "gradient")
    @test !_all_passed(check_consistency(Functional((u, θ) -> ∫(f * u)dΩ, S; derivative = DJ(1.05), adjoint = :self, fields = θ, scale = 1.0)), "gradient")
    @test !_all_passed(check_consistency(C; min_rate = 2.5), "gradient")
    n0 = length(p.ctxs)
    r = check_consistency(C; ctx = :my_context)
    @test length(p.ctxs) > n0 && all(c -> c[2] === :my_context, p.ctxs[n0+1:end])   # every solve of the call
    # sensitivity of the verdict (documented): 1 % in DJform is found
    @test !_all_passed(check_consistency(Functional((u, θ) -> ∫(f * u)dΩ, S; derivative = DJ(1.01), adjoint = :self, fields = θ, scale = 1.0)), "gradient")
    # a saturated projection: J does not change along δ, nothing is tested
    set_beta!(p.H, 512.0)
    r = check_consistency(C)
    x = only(_results(r, "gradient"))
    @test !x.passed && occursin("not tested", x.message)
  end

  @testset "A missing gradient is not hidden by a large constant" begin
    q = filter_problem(symmetric_filter)
    dΩ = q.dΩ
    F = Functional(θ -> ∫(1.0e8 + θ.ρ)dΩ, q.θ; derivative = θ -> (δ -> ∫(0.0 * δ)dΩ), scale = 1.0)
    @test !_all_passed(check_consistency(F), "gradient")
    G = Functional(θ -> ∫(1.0e8 + θ.ρ)dΩ, q.θ; derivative = θ -> (δ -> ∫(δ)dΩ), scale = 1.0)
    @test _all_passed(check_consistency(G), "gradient")
  end

  @testset "Number of solves" begin
    p = state_problem()
    (; θ, S, A, f, dk, dΩ, dΓ, nforward, nadjoint) = p
    C = Functional((u, θ) -> ∫(f * u)dΩ, S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = :self, fields = θ, scale = 1.0)
    B = Functional((u, θ) -> ∫(u)dΓ, S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = A, fields = θ, scale = 1.0)
    functional_gradient!(zeros(length(p.z)), B, nothing)
    f0, a0 = nforward[], nadjoint[]
    @test check_consistency(C, B; steps = 5).passed
    @test nforward[] - f0 <= 5 + 4                  # shared by both functionals
    @test nadjoint[] - a0 <= 4
  end

  @testset "Failures of the state and errors in user code" begin
    for failfrom in (1, 3)                          # fails at the current design, or at a perturbed one
      p = state_problem()
      (; z, θ, f, dk, dΩ) = p
      cnt = Ref(0)
      S2 = StateOperator(p.S.model; depends = θ, solve = (m, ctx) -> (cnt[] += 1; (solve!(m; Assembly = true), cnt[] < failfrom)))
      C = Functional((u, θ) -> ∫(f * u)dΩ, S2; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = :self, fields = θ, scale = :initial)
      z0 = copy(get_values(z))
      r = check_consistency(C)
      @test !r.passed && !_all_passed(r, "gradient")
      @test get_values(z) == z0 && isnan(C.scale[])
    end
    p = state_problem()
    (; z, θ, S, f, dk, dΩ) = p
    cnt = Ref(0)
    C = Functional((u, θ) -> (cnt[] += 1; cnt[] == 4 && error("boom"); ∫(f * u)dΩ), S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = :self, fields = θ, scale = :initial)
    z0 = copy(get_values(z))
    @test_throws ErrorException check_consistency(C)
    @test get_values(z) == z0 && isnan(C.scale[])
  end

  @testset "Several physical fields, infinite bounds" begin
    model = CartesianDiscreteModel((0, 1, 0, 1), (5, 5))
    dΩ = Measure(Triangulation(model), 2)
    V0 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
    z = OptVariables(ρ = FieldVariable(V0; bounds = (0.0, Inf), init = [0.5 + 0.3 * sin(i) for i in 1:num_free_dofs(V0)]))
    x = DesignFields(z; ρ = Identity())
    θ = PhysicalFields(x; ρ = Heaviside(β = 2.0, η = 0.5), E = :ρ => Identity())
    F = Functional(θ -> ∫(θ.ρ * θ.E)dΩ, θ; derivative = θ -> (ρ = δ -> ∫(θ.E * δ)dΩ, E = δ -> ∫(θ.ρ * δ)dΩ), scale = 1.0)
    r = check_consistency((energy = F,))
    @test r.passed
    @test _all_passed(r, "design chain of θ.ρ") && _all_passed(r, "design chain of θ.E")
  end

  @testset "Tangents: nonlinear residual, Dirichlet values, labels" begin
    p = state_problem(; gD = 1.0, adjoint_space = :homogeneous)
    (; θ, S, A, dk, dΩ, dΓ) = p
    B = Functional((u, θ) -> ∫(u)dΓ, S; derivative = (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), adjoint = A, fields = θ, scale = 1.0)
    res(u, v) = ∫((1 + u * u) * ∇(u) ⋅ ∇(v))dΩ
    jac(c) = (u, du, v) -> ∫((1 + u * u) * ∇(du) ⋅ ∇(v) + c * u * du * ∇(u) ⋅ ∇(v))dΩ
    r = check_consistency(B; tangents = (S => (res, jac(2.0)), S => (res, jac(1.9))))
    @test only(_results(r, "tangent 1")).passed
    @test !only(_results(r, "tangent 2")).passed
  end

  @testset "Under-declared state reached through depends" begin
    p = state_problem()
    (; θ, U, V, bc, a, l, dΩ) = p
    # S1 reads θ but declares nothing; S2 reads the state of S1 and declares it
    for deps1 in ((), θ)
      S1 = StateOperator(StaticLinearModel(l, a, U, V, bc; ls = LUSolver()); depends = deps1,
                         solve = (m, ctx) -> solve!(m; Assembly = true))
      u1 = get_state(S1)
      S2 = StateOperator(StaticLinearModel(v -> ∫(u1 * v)dΩ, (u, v) -> ∫(∇(u) ⋅ ∇(v))dΩ, U, V, bc; ls = LUSolver());
                         depends = (θ, S1), solve = (m, ctx) -> solve!(m; Assembly = true))
      F = Functional((u, θ) -> ∫(u)dΩ, S2; derivative = (u, w, θ) -> (δ -> ∫(0.0 * δ)dΩ), adjoint = :self, fields = θ, scale = 1.0)
      r = check_consistency(F)
      @test _all_passed(r, "dependencies") == (deps1 !== ())
    end
  end

  @testset "Nearly linear functional: inconclusive, not passed" begin
    q = filter_problem(symmetric_filter)
    dΩ = q.dΩ
    n = length(q.z)
    z = OptVariables(ρ = FieldVariable(get_fe_space(q.x.ρ); bounds = (0.0, 1.0), init = 0.5))
    x = DesignFields(z; ρ = Identity())
    θ = PhysicalFields(x; ρ = Identity())
    ε = 3e-6
    F = Functional(θ -> ∫(θ.ρ + ε * θ.ρ * θ.ρ)dΩ, θ; derivative = θ -> (δ -> ∫((1 + 2ε * θ.ρ) * δ)dΩ), scale = 1.0)
    r = check_consistency(F; direction = fill(0.1, n), h0 = 1e-2)
    g = only(_results(r, "gradient"))
    @test !g.passed && occursin("inconclusive", g.message)
    @test _all_passed(check_consistency(F; direction = fill(1.0, n), h0 = 0.4), "gradient")   # larger steps
  end

  @testset "Two physical fields through one filter" begin
    q = filter_problem(symmetric_filter)
    Sf = HyperShape.Variables.filter_operators(q.θ)[1]
    dΩ = q.dΩ
    θ2 = PhysicalFields(q.x; ρ = Heaviside(β = 3.0, η = 0.4) ∘ HelmholtzFilter(Sf, dΩ), E = :ρ => HelmholtzFilter(Sf, dΩ))
    F = Functional(θ -> ∫(θ.ρ * θ.E)dΩ, θ2; derivative = θ -> (ρ = δ -> ∫(θ.E * δ)dΩ, E = δ -> ∫(θ.ρ * δ)dΩ), scale = 1.0)
    r = check_consistency(F)
    @test r.passed
    @test length(_results(r, "design chain")) == 2 && length(_results(r, "symmetry")) == 1
  end
end
