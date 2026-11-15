using Gridap
using Gridap.FESpaces
using HyperFEM
using HyperShape
using HyperShape.Variables: get_values, set_values!, version
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
  θ = PhysicalFields(x; ρ = Heaviside(β = 2.0, η = 0.5) ∘ HelmholtzFilter(Sf; measure = dΩ))
  F = Functional(θ -> ∫(θ.ρ * θ.ρ)dΩ, θ -> (δ -> ∫(2 * θ.ρ * δ)dΩ), θ; scale = 1.0)
  (; z, x, θ, F, dΩ)
end

symmetric_filter(x, dΩ) = (u, v) -> ∫(0.01 * ∇(u) ⋅ ∇(v) + u * v)dΩ

@testset "Checks" begin

  @testset "Consistent functionals pass ($(design))" for design in (:P0, :P1)
    p = state_problem(; design)
    (; z, θ, S, A, f, dk, dΩ, dΓ, a, l) = p
    C = Functional((u, θ) -> ∫(f * u)dΩ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), S, S, θ; scale = :initial)
    B = Functional((u, θ) -> ∫(u)dΓ, (u, w, θ) -> (δ -> ∫(-(dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), S, A, θ; scale = 2.0)
    Vol = Functional(θ -> ∫(θ.ρ)dΩ, θ -> (δ -> ∫(δ)dΩ), θ; scale = 1.0)
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
    C = Functional((u, θ) -> ∫(f * u)dΩ, (u, w, θ) -> (δ -> ∫(-2 * (dk ∘ θ.ρ) * ∇(u) ⋅ ∇(w) * δ)dΩ), S, S, θ; scale = 1.0)
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
    Vol = Functional(θ -> ∫(θ.ρ)dΩ, θ -> (δ -> ∫(δ)dΩ), p.θ; scale = 1.0)
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
end
