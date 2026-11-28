using Gridap
using Gridap.FESpaces
using HyperFEM
using HyperFEMKernels
using HyperShape
using HyperShape.Variables: update_fields!, design_gradient!, set_values!, get_values, get_bounds, changed!,
                            set_beta!, block_range, TanhValue, state_version
using LinearAlgebra
using Test

function filter_problem(; n = 8, r = 0.1)
  model = CartesianDiscreteModel((0, 1, 0, 1), (n, n))
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  V0 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
  V1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
  Vf = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
  Uf = TrialFESpace(Vf)
  af(u, v) = ∫(r^2 * ∇(u) ⋅ ∇(v) + u * v)dΩ
  nsolves = Ref(0)
  solvef = (m, b, ctx, stale) -> (nsolves[] += 1; solve!(m, b; Assembly = stale))
  Sf = LinearOperator(StaticLinearModel(af, Uf, Vf, NothingBC(); ls = LUSolver()); depends = (), solve = solvef)
  (; model, Ω, dΩ, V0, V1, Vf, Uf, af, Sf, nsolves)
end

# smooth coefficient for the test functionals
cfun(x) = 1 + x[1] + 2 * x[2]^2

# Taylor test: e(h) = |J(z + h d) - J(z) - h g⋅d| must decrease with order 2
function taylor_rates(J, g, z, d)
  z0 = copy(get_values(z))
  J0 = J()
  gd = dot(g, d)
  errs = Float64[]
  for h in (1e-2, 5e-3, 2.5e-3, 1.25e-3)
    set_values!(z, z0 .+ h .* d)
    push!(errs, abs(J() - J0 - h * gd))
  end
  set_values!(z, z0)
  [log2(errs[i] / errs[i+1]) for i in 1:length(errs)-1]
end

@testset "Variables" begin

  p = filter_problem()
  ρ0 = [0.2 + 0.6 * abs(sin(3.1i)) for i in 1:num_free_dofs(p.V0)]
  ρ1 = [0.2 + 0.6 * abs(cos(1.7i)) for i in 1:num_free_dofs(p.V1)]

  @testset "OptVariables" begin
    z = OptVariables(a = FieldVariable(p.V0; bounds = (0.0, 1.0), init = ρ0),
                     b = FieldVariable(p.V1; bounds = (-1.0, 2.0), init = 0.5))
    n0, n1 = num_free_dofs(p.V0), num_free_dofs(p.V1)
    @test length(z) == n0 + n1
    @test block_range(z, :a) == 1:n0 && block_range(z, :b) == (n0+1):(n0+n1)
    @test get_values(z)[1:n0] == ρ0 && all(==(0.5), get_values(z)[n0+1:end])
    lb, ub = get_bounds(z)
    @test all(==(0.0), lb[1:n0]) && all(==(-1.0), lb[n0+1:end]) && all(==(2.0), ub[n0+1:end])
    v = state_version(z)
    set_values!(z, zeros(length(z)))
    @test state_version(z) == v + 1
    changed!(z)
    @test state_version(z) == v + 2
    @test_throws DimensionMismatch set_values!(z, zeros(3))
    @test_throws ArgumentError OptVariables()
    @test_throws ArgumentError OptVariables(a = p.V0)
    @test_throws ArgumentError FieldVariable(p.V0; bounds = (1.0, 1.0), init = 1.0)
    @test_throws ArgumentError FieldVariable(p.V0; bounds = (0.0, 1.0), init = 1.5)
    @test_throws DimensionMismatch FieldVariable(p.V0; bounds = (0.0, 1.0), init = [0.5])
    @test_throws r"bounds is mandatory" FieldVariable(p.V0; init = 0.5)
    @test_throws r"init is mandatory" FieldVariable(p.V0; bounds = (0.0, 1.0))
    Vd = TestFESpace(p.model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1, dirichlet_tags = "boundary")
    @test_throws ArgumentError FieldVariable(Vd; bounds = (0.0, 1.0), init = 0.5)
  end

  @testset "DesignFields" begin
    z = OptVariables(a = FieldVariable(p.V0; bounds = (0.0, 1.0), init = ρ0),
                     b = FieldVariable(p.V1; bounds = (0.0, 1.0), init = ρ1))
    x = DesignFields(z; b = Identity(), a = Identity())
    @test get_free_dof_values(x.a) == ρ0 && get_free_dof_values(x.b) == ρ1
    @test propertynames(x) == (:a, :b)
    fa = x.a
    vx = state_version(x)
    update_fields!(x, nothing)
    @test state_version(x) == vx                         # nothing changed
    w = copy(get_values(z)); w[1] = 0.9
    set_values!(z, w)
    update_fields!(x, nothing)
    @test state_version(x) == vx + 1
    @test x.a === fa && get_free_dof_values(x.a)[1] == 0.9    # same object, updated in place
    @test_throws ArgumentError DesignFields(z; a = Identity())
    @test_throws ArgumentError DesignFields(z; a = Identity(), b = Identity(), c = Identity())
    @test_throws ArgumentError DesignFields(z; a = Identity(), b = TanhProjection(β = 1.0, η = 0.5))
    @test_throws r"give a map for every block .*, or none" DesignFields(z; a = Identity())
    # the identity implied (decision of Jesús, 2026-11-28): DesignFields(z), PhysicalFields(x; E = :a), PhysicalFields(x)
    xi = DesignFields(z)
    @test propertynames(xi) == (:a, :b) && get_free_dof_values(xi.a) == get_free_dof_values(x.a)
    θi, θe = PhysicalFields(xi), PhysicalFields(xi; a = Identity(), b = Identity())
    @test propertynames(θi) == (:a, :b) && typeof(θi) == typeof(θe)
    θs = PhysicalFields(xi; E = :a, b = :b)
    @test propertynames(θs) == (:E, :b) && typeof(θs) == typeof(PhysicalFields(xi; E = :a => Identity(), b = :b => Identity()))
    @test getfield(θs, :sources) == (1, 2)
    @test_throws ArgumentError PhysicalFields(xi; E = :c)
    @test_throws r"give the measure of the domain, Volume" Volume(θi; max_fraction = 0.5)
    @test_throws r"give the measure of the domain, Volume" Volume(xi; max_fraction = 0.5, field = :a)
    @test typeof(PhysicalFields(xi; backend = :lazy)) == typeof(PhysicalFields(xi; a = Identity(), b = Identity(), backend = :lazy))
  end

  @testset "Helmholtz filter = direct solve ($(name))" for (name, V, ρ) in (("P0", p.V0, ρ0), ("P1", p.V1, ρ1))
    z, x, θ = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ,
                                  map = HelmholtzFilter(p.Sf, p.dΩ))
    xh = FEFunction(V, ρ)
    dΩ = p.dΩ
    ref = solve(LUSolver(), AffineFEOperator(p.af, v -> ∫(xh * v)dΩ, p.Uf, p.Vf))
    @test θ.ρ isa FEFunction
    @test maximum(abs, get_free_dof_values(θ.ρ) .- get_free_dof_values(ref)) < 1e-12
  end

  @testset "Transpose of the filter ($(name))" for (name, V, ρ) in (("P0", p.V0, ρ0), ("P1", p.V1, ρ1))
    # J(z) = ∫(y F(z)) dΩ is linear in z with J(0) = 0, so J(z) = g⋅z exactly
    z, x, θ = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ,
                                  map = HelmholtzFilter(p.Sf, p.dΩ))
    yh = interpolate(cfun, p.Uf)
    dΩ = p.dΩ
    g = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(yh * δ)dΩ,), nothing)
    J = sum(∫(yh * θ.ρ)dΩ)
    @test dot(g, get_values(z)) ≈ J rtol = 1e-12
  end

  @testset "Taylor test: projection ∘ filter ($(name))" for (name, V, ρ) in (("P0", p.V0, ρ0), ("P1", p.V1, ρ1))
    H = TanhProjection(β = 4.0, η = 0.5)
    z, x, θ = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ,
                                  map = H ∘ HelmholtzFilter(p.Sf, p.dΩ))
    c = CellField(cfun, p.Ω)
    dΩ = p.dΩ
    J() = (update_fields!(θ, nothing); sum(∫(c * θ.ρ * θ.ρ)dΩ))
    g = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(2 * c * θ.ρ * δ)dΩ,), nothing)
    d = [sin(0.7i) for i in 1:length(z)]
    rates = taylor_rates(J, g, z, d)
    @test all(r -> 1.9 < r < 2.1, rates)
    # same test after changing β (continuation)
    set_beta!(H, 12.0)
    g = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(2 * c * θ.ρ * δ)dΩ,), nothing)
    rates = taylor_rates(J, g, z, d)
    @test all(r -> 1.9 < r < 2.1, rates)
  end

  @testset "Composition order: H ∘ F filters first" begin
    H = TanhProjection(β = 6.0, η = 0.4)
    z = OptVariables(ρ = FieldVariable(p.V0; bounds = (0.0, 1.0), init = ρ0))
    x = DesignFields(z; ρ = Identity())
    θ = PhysicalFields(x; ρ = H ∘ HelmholtzFilter(p.Sf, p.dΩ), ρf = :ρ => HelmholtzFilter(p.Sf, p.dΩ))
    dΩ = p.dΩ
    hv = TanhValue(H)
    @test sum(∫(θ.ρ)dΩ) ≈ sum(∫(Operation(hv)(θ.ρf))dΩ) rtol = 1e-14
    @test !(sum(∫(θ.ρ)dΩ) ≈ sum(∫(Operation(hv)(x.ρ))dΩ))
  end

  @testset "Several physical fields from one design field; two design blocks" begin
    H = TanhProjection(β = 3.0, η = 0.5)
    z = OptVariables(a = FieldVariable(p.V0; bounds = (0.0, 1.0), init = ρ0),
                     b = FieldVariable(p.V1; bounds = (0.0, 1.0), init = ρ1))
    x = DesignFields(z; a = Identity(), b = Identity())
    θ = PhysicalFields(x; ρ = :a => H ∘ HelmholtzFilter(p.Sf, p.dΩ),
                          E = :a => HelmholtzFilter(p.Sf, p.dΩ),
                          b = Identity())
    c = CellField(cfun, p.Ω)
    dΩ = p.dΩ
    J() = (update_fields!(θ, nothing); sum(∫(c * θ.ρ * θ.ρ + θ.E * θ.E * θ.E + c * θ.b * θ.ρ)dΩ))
    forms = (ρ = δ -> ∫((2 * c * θ.ρ + c * θ.b) * δ)dΩ, E = δ -> ∫(3 * θ.E * θ.E * δ)dΩ, b = δ -> ∫(c * θ.ρ * δ)dΩ)
    g = design_gradient!(zeros(length(z)), θ, forms, nothing)
    d = [cos(1.3i) for i in 1:length(z)]
    rates = taylor_rates(J, g, z, d)
    @test all(r -> 1.9 < r < 2.1, rates)
    # a design field that no physical field uses
    @test_throws ArgumentError PhysicalFields(x; ρ = HelmholtzFilter(p.Sf, p.dΩ))
    @test_throws ArgumentError PhysicalFields(x; ρ = :c => Identity(), b = Identity())
    @test_throws ArgumentError PhysicalFields(x; a = 1.0, b = Identity())
    @test_throws ArgumentError design_gradient!(zeros(length(z)), θ, (q = δ -> ∫(δ)dΩ,), nothing)
  end

  @testset "Versions and number of filter solves" begin
    H = TanhProjection(β = 2.0, η = 0.5)
    z, x, θ = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0,
                                  map = H ∘ HelmholtzFilter(p.Sf, p.dΩ))
    n = p.nsolves[]
    vθ = state_version(θ)
    update_fields!(θ, nothing)
    @test p.nsolves[] == n && state_version(θ) == vθ          # nothing changed: no solve
    changed!(z)
    update_fields!(θ, nothing)
    @test p.nsolves[] == n + 1 && state_version(θ) == vθ + 1  # one filter solve
    set_beta!(H, 4.0)
    update_fields!(θ, nothing)
    @test p.nsolves[] == n + 1 && state_version(θ) == vθ + 2  # β counts as a change; the projection is lazy: no solve
    set_beta!(H, 4.0)
    update_fields!(θ, nothing)
    @test state_version(θ) == vθ + 2                           # same β: no change
    dΩ = p.dΩ
    g1 = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(δ)dΩ,), nothing)
    @test p.nsolves[] == n + 2                           # one transposed solve
    g2 = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(2 * δ)dΩ,), nothing)
    @test p.nsolves[] == n + 3 && g2 ≈ 2 .* g1           # another form, same design: one solve, no update
    # projection before the filter: a new β changes the input of the filter
    H2 = TanhProjection(β = 2.0, η = 0.5)
    z2, x2, θ2 = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0,
                                     map = HelmholtzFilter(p.Sf, p.dΩ) ∘ H2)
    n = p.nsolves[]
    set_beta!(H2, 5.0)
    update_fields!(θ2, nothing)
    @test p.nsolves[] == n + 1
    @test_throws ArgumentError TanhProjection(β = 0.0, η = 0.5)
    @test_throws ArgumentError TanhProjection(β = 1.0, η = 1.0)
    @test_throws ArgumentError set_beta!(H, -1.0)
    @test_throws ArgumentError (H.η = 0.3)
    @test_throws ArgumentError (H.β = 3.0)
    @test_throws DimensionMismatch design_gradient!(zeros(3), θ, (ρ = δ -> ∫(δ)dΩ,), nothing)
    @test_throws ArgumentError θ.foo
    @test_throws ArgumentError x.foo
  end

  @testset "Values of z are checked" begin
    z, x, θ = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0, map = Identity())
    v = state_version(z)
    w = copy(get_values(z)); w[3] = NaN
    @test_throws ArgumentError set_values!(z, w)
    w[3] = 1.5
    @test_throws ArgumentError set_values!(z, w)
    get_values(z)[2] = -0.1
    @test_throws ArgumentError changed!(z)
    @test state_version(z) == v
  end

  @testset "Filter with homogeneous Dirichlet values" begin
    Vd = TestFESpace(p.model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1, dirichlet_tags = "boundary")
    Ud = TrialFESpace(Vd, 0.0)
    af, dΩ = p.af, p.dΩ
    Sd = LinearOperator(StaticLinearModel(af, Ud, Vd, NothingBC(); ls = LUSolver()); depends = (),
                        solve = (m, b, ctx, stale) -> solve!(m, b; Assembly = stale))
    z, x, θ = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0,
                                  map = TanhProjection(β = 3.0, η = 0.5) ∘ HelmholtzFilter(Sd, dΩ))
    xh = FEFunction(p.V0, ρ0)
    ref = solve(LUSolver(), AffineFEOperator(af, v -> ∫(xh * v)dΩ, Ud, Vd))
    filtered = PhysicalFields(x; ρ = HelmholtzFilter(Sd, dΩ))
    @test maximum(abs, get_free_dof_values(filtered.ρ) .- get_free_dof_values(ref)) < 1e-12
    c = CellField(cfun, p.Ω)
    J() = (update_fields!(θ, nothing); sum(∫(c * θ.ρ * θ.ρ)dΩ))
    g = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(2 * c * θ.ρ * δ)dΩ,), nothing)
    @test all(r -> 1.9 < r < 2.1, taylor_rates(J, g, z, [sin(0.7i) for i in 1:length(z)]))
    # non-homogeneous values are rejected (the load vector carries no lift)
    Un = TrialFESpace(Vd, 1.0)
    Sn = LinearOperator(StaticLinearModel(af, Un, Vd, NothingBC(); ls = LUSolver()); depends = (),
                        solve = (m, b, ctx, stale) -> solve!(m, b; Assembly = stale))
    @test_throws ArgumentError PhysicalFields(x; ρ = HelmholtzFilter(Sn, dΩ))
  end

  @testset "Transpose with independent vectors ($(name))" for (name, V, ρ) in (("P0", p.V0, ρ0), ("P1", p.V1, ρ1))
    # g is the gradient of z ↦ ∫(y F(z)) dΩ at one design; ∫(y F(z₂)) dΩ = g⋅z₂ for another design z₂
    z, x, θ = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ,
                                  map = HelmholtzFilter(p.Sf, p.dΩ))
    yh = interpolate(x -> sin(4x[1]) * cos(3x[2]), p.Uf)
    dΩ = p.dΩ
    g = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(yh * δ)dΩ,), nothing)
    z2 = [0.1 + 0.8 * abs(cos(2.3i)) for i in 1:length(z)]
    set_values!(z, z2)
    update_fields!(θ, nothing)
    @test sum(∫(yh * θ.ρ)dΩ) ≈ dot(g, z2) rtol = 1e-12
  end

  @testset "Taylor tests of other chains ($(name), $(label))" for (name, V, ρ) in (("P0", p.V0, ρ0), ("P1", p.V1, ρ1)),
                                                                     (label, mk) in (("H", (F, H, H2) -> H),
                                                                                     ("F ∘ H", (F, H, H2) -> F ∘ H),
                                                                                     ("F ∘ F", (F, H, H2) -> F ∘ F),
                                                                                     ("H ∘ H ∘ F", (F, H, H2) -> H2 ∘ H ∘ F))
    F = HelmholtzFilter(p.Sf, p.dΩ)
    map = mk(F, TanhProjection(β = 3.0, η = 0.45), TanhProjection(β = 2.0, η = 0.55))
    z, x, θ = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ, map = map)
    c = CellField(cfun, p.Ω)
    dΩ = p.dΩ
    J() = (update_fields!(θ, nothing); sum(∫(c * θ.ρ * θ.ρ)dΩ))
    g = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(2 * c * θ.ρ * δ)dΩ,), nothing)
    @test all(r -> 1.9 < r < 2.1, taylor_rates(J, g, z, [sin(0.7i) for i in 1:length(z)]))
  end

  @testset "Two design blocks, each filtered and projected" begin
    z = OptVariables(a = FieldVariable(p.V0; bounds = (0.0, 1.0), init = ρ0),
                     b = FieldVariable(p.V1; bounds = (0.0, 1.0), init = ρ1))
    x = DesignFields(z; a = Identity(), b = Identity())
    θ = PhysicalFields(x; a = TanhProjection(β = 3.0, η = 0.5) ∘ HelmholtzFilter(p.Sf, p.dΩ),
                          b = TanhProjection(β = 5.0, η = 0.4) ∘ HelmholtzFilter(p.Sf, p.dΩ))
    c = CellField(cfun, p.Ω)
    dΩ = p.dΩ
    J() = (update_fields!(θ, nothing); sum(∫(c * θ.a * θ.b + θ.b * θ.b)dΩ))
    g = design_gradient!(zeros(length(z)), θ, (a = δ -> ∫(c * θ.b * δ)dΩ, b = δ -> ∫((c * θ.a + 2 * θ.b) * δ)dΩ), nothing)
    @test all(r -> 1.9 < r < 2.1, taylor_rates(J, g, z, [cos(0.9i) for i in 1:length(z)]))
  end

  @testset "Memory per update does not grow with the mesh" begin
    dΩs = Dict{Int,Any}()
    bytes = map((8, 16)) do n
      q = filter_problem(; n)
      z, x, θ = single_field_design(:ρ, q.V0; bounds = (0.0, 1.0), init = 0.5,
                                    map = TanhProjection(β = 3.0, η = 0.5) ∘ HelmholtzFilter(q.Sf, q.dΩ))
      dΩ = q.dΩ
      forms = (ρ = δ -> ∫(δ)dΩ,)
      g = zeros(length(z))
      for _ in 1:2
        changed!(z); update_fields!(θ, nothing); design_gradient!(g, θ, forms, nothing)
      end
      changed!(z)
      up = @allocated update_fields!(θ, nothing)
      gr = @allocated design_gradient!(g, θ, forms, nothing)
      none = @allocated update_fields!(θ, nothing)
      (up, gr, none)
    end
    @test bytes[1][3] == 0 && bytes[2][3] == 0            # nothing changed: no allocation
    # Gridap rebuilds the lazy forms at each assembly: a fixed amount, independent of the number of cells
    @test bytes[2][1] <= 1.2 * bytes[1][1] + 1024
    @test bytes[2][2] <= 1.2 * bytes[1][2] + 1024
  end

  @testset "LinearOperator: depends and stale" begin
    z = OptVariables(ρ = FieldVariable(p.V0; bounds = (0.0, 1.0), init = 0.5))
    flags = Bool[]
    op = LinearOperator(StaticLinearModel(p.af, p.Uf, p.Vf, NothingBC(); ls = LUSolver()); depends = (z,),
                        solve = (m, b, ctx, stale) -> (push!(flags, stale); solve!(m, b; Assembly = stale)))
    b = ones(num_free_dofs(p.Vf))
    y = zeros(num_free_dofs(p.Vf))
    HyperShape.Operators.solve_linear!(y, op, b, nothing)
    changed!(z)
    HyperShape.Operators.solve_linear!(y, op, b, nothing)
    HyperShape.Operators.solve_linear!(y, op, b, nothing)
    @test flags == [false, true, false]
    # a solve that throws does not consume the change: the next call is still stale
    failing = Ref(true)
    op2 = LinearOperator(op.model; depends = (z,),
                         solve = (m, b, ctx, stale) -> (push!(flags, stale); failing[] && error("no convergence");
                                                        solve!(m, b; Assembly = stale)))
    changed!(z)
    @test_throws ErrorException HyperShape.Operators.solve_linear!(y, op2, b, nothing)
    failing[] = false
    HyperShape.Operators.solve_linear!(y, op2, b, nothing)
    @test flags[end-1:end] == [true, true]
    @test LinearOperator(op.model; depends = z, solve = op.solve).depends == (z,)   # one object is accepted
    @test_throws ArgumentError LinearOperator(op.model; depends = (1.0,), solve = op.solve)
    @test LinearOperator(op.model; solve = op.solve).depends === ()   # constant matrix by default
    bad = LinearOperator(op.model; depends = (), solve = (m, b, ctx, stale) -> zeros(3))
    @test_throws DimensionMismatch HyperShape.Operators.solve_linear!(y, bad, b, nothing)
  end

  @testset "Weak forms capture θ: lazy and compiled backends" begin
    H = TanhProjection(β = 4.0, η = 0.5)
    z, x, θ = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0,
                                  map = H ∘ HelmholtzFilter(p.Sf, p.dΩ))
    Ω, dΩ = p.Ω, p.dΩ
    bc = BoundaryConditions(BoundaryCondition("boundary", 0.0))
    V = TestFESpace(Ω, ReferenceFE(lagrangian, Float64, 1), bc, conformity = :H1)
    U = TrialFESpace(V, bc, 1.0)
    a(u, v) = ∫((1e-3 + θ.ρ) * ∇(u) ⋅ ∇(v))dΩ
    l(v) = ∫(v)dΩ
    models = (lazy = StaticLinearModel(l, a, U, V, bc; ls = LUSolver()),
              compiled = StaticLinearModel(l, a, U, V, bc; ls = LUSolver(), backend = :compiled))
    reference() = get_free_dof_values(solve(LUSolver(), AffineFEOperator(a, l, U, V)))
    for (k, m) in pairs(models)
      @test maximum(abs, solve!(m; Assembly = true) .- reference()) < 1e-10
    end
    w = copy(get_values(z)); w[1:2:end] .= 0.95
    set_values!(z, w)
    update_fields!(θ, nothing)
    set_beta!(H, 9.0)
    xref = reference()
    for (k, m) in pairs(models)
      @test maximum(abs, solve!(m; Assembly = true) .- xref) < 1e-10
    end
  end

  @testset "Compiled and lazy backends give the same fields and gradients ($(name), $(label))" for (name, V, ρ) in (("P0", p.V0, ρ0), ("P1", p.V1, ρ1)),
                                                                                                           (label, mk) in (("H ∘ F", (F, H) -> H ∘ F), ("F ∘ H", (F, H) -> F ∘ H), ("H", (F, H) -> H))
    c = CellField(cfun, p.Ω)
    dΩ = p.dΩ
    out = map((:compiled, :lazy)) do backend
      H = TanhProjection(β = 3.0, η = 0.45)
      z, x, θ = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ, backend,
                                    map = mk(HelmholtzFilter(p.Sf, dΩ; backend), H))
      g1 = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(2 * c * θ.ρ * δ)dΩ,), nothing; key = 1)
      # another quantity under another key, then the first one again after changing z and β
      g2 = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(δ)dΩ,), nothing; key = 2)
      set_values!(z, clamp.(get_values(z) .+ 0.05, 0, 1))
      set_beta!(H, 6.0)
      g3 = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(2 * c * θ.ρ * δ)dΩ,), nothing; key = 1)
      # a form of another structure under a used key: compiled again, still correct
      g4 = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(c * δ)dΩ,), nothing; key = 1)
      (g1, g2, g3, g4, sum(∫(θ.ρ)dΩ))
    end
    for k in 1:4
      @test maximum(abs, out[1][k] .- out[2][k]) <= 1e-12 * maximum(abs, out[2][k])
    end
    @test out[1][5] ≈ out[2][5] rtol = 1e-13
  end

  @testset "Compiled forms: captured values, boundary forms and a bounded cache" begin
    dΩ = p.dΩ
    dΓ = Measure(BoundaryTriangulation(p.model), 2)
    Vs = TestFESpace(p.model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    out = map((:compiled, :lazy)) do backend
      H = TanhProjection(β = 3.0, η = 0.45)
      z, x, θ = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0, backend,
                                    map = H ∘ HelmholtzFilter(p.Sf, dΩ; backend))
      gs = Vector{Float64}[]
      for a in (1.0, 2.5)                                  # a number captured by the form
        push!(gs, design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(a * θ.ρ * δ)dΩ,), nothing; key = 1))
      end
      for k in (1.0, 3.0)                                  # a new FE function at every call
        uh = FEFunction(Vs, [sin(k * i) for i in 1:num_free_dofs(Vs)])
        push!(gs, design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(uh * uh * δ + ∇(uh) ⋅ ∇(uh) * δ)dΩ,), nothing; key = 2))
      end
      push!(gs, design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(δ)dΓ + ∫(θ.ρ * δ)dΩ,), nothing; key = 3))   # boundary term
      for k in 1:40                                        # many quantities: the cache stays bounded
        design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(k * δ)dΩ,), nothing; key = 100 + k)
      end
      push!(gs, design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(2.5 * θ.ρ * δ)dΩ,), nothing; key = 1))      # an evicted key
      gs, θ
    end
    for k in eachindex(out[1][1])
      @test maximum(abs, out[1][1][k] .- out[2][1][k]) <= 1e-12 * maximum(abs, out[2][1][k])
    end
    @test out[1][1][6] ≈ out[1][1][2]
    θc = out[1][2]
    assems = getfield(θc, :assems)
    @test length(assems[1].ops) <= assems[1].capacity[]
    filter_stage = getfield(θc, :chains)[1][1]
    fa = filter_stage.assem
    @test length(fa.ops) <= fa.capacity[] == HyperShape.Operators.MAX_COMPILED_FORMS + 1   # key 1 came back once
    @test any(k -> isequal(k, :forward), filter_stage.assem.keys)   # the load vector of the filter is never dropped
    # every cached operator shares the tables of the first one
    @test all(op -> op.fields === assems[1].ops[1].fields && op.colors === assems[1].ops[1].colors, assems[1].ops)
  end

  @testset "Compiled forms: more quantities than slots stop recompiling" begin
    dΩ = p.dΩ
    z, x, θ = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0, map = HelmholtzFilter(p.Sf, dΩ))
    a = getfield(θ, :chains)[1][1].assem                    # the forms are accumulated at the filter
    nk = HyperShape.Operators.MAX_COMPILED_FORMS + 4
    for _ in 1:2, k in 1:nk                                 # a cycle longer than the cache: every key comes back
      design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(k * θ.ρ * δ)dΩ,), nothing; key = k)
    end
    nev = length(a.evicted)
    for k in 1:nk
      design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(k * θ.ρ * δ)dΩ,), nothing; key = k)
    end
    @test a.capacity[] >= nk && length(a.ops) == length(Set(a.keys))
    @test length(a.evicted) == nev                          # nothing dropped in the third cycle
    @test all(k -> any(isequal((k, 1)), a.keys), 1:nk)   # (key, physical field)
  end

  @testset "Taylor test with the lazy backend ($(name))" for (name, V, ρ) in (("P0", p.V0, ρ0), ("P1", p.V1, ρ1))
    z, x, θ = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ, backend = :lazy,
                                  map = TanhProjection(β = 4.0, η = 0.5) ∘ HelmholtzFilter(p.Sf, p.dΩ; backend = :lazy))
    c = CellField(cfun, p.Ω)
    dΩ = p.dΩ
    J() = (update_fields!(θ, nothing); sum(∫(c * θ.ρ * θ.ρ)dΩ))
    g = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(2 * c * θ.ρ * δ)dΩ,), nothing)
    @test all(r -> 1.9 < r < 2.1, taylor_rates(J, g, z, [sin(0.7i) for i in 1:length(z)]))
  end

  @testset "Backend options" begin
    zb = OptVariables(ρ = FieldVariable(p.V0; bounds = (0.0, 1.0), init = ρ0))
    @test_throws ArgumentError PhysicalFields(DesignFields(zb; ρ = Identity()); ρ = Identity(), backend = Identity())
    @test_throws ArgumentError HelmholtzFilter(p.Sf, p.dΩ; backend = :gpu)
    z = OptVariables(ρ = FieldVariable(p.V0; bounds = (0.0, 1.0), init = ρ0))
    x = DesignFields(z; ρ = Identity())
    @test_throws ArgumentError PhysicalFields(x; ρ = Identity(), backend = :gpu)
    @test_throws ArgumentError single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = 0.5, map = Identity(), backend = :gpu)
    F = HelmholtzFilter(p.Sf, p.dΩ)
    @test F.backend === :compiled                                    # the default
    @test HelmholtzFilter(p.Sf, p.dΩ; backend = :lazy).backend === :lazy
  end

  @testset "Short constructor of the filter ($(name))" for (name, V, ρ) in (("P0", p.V0, ρ0), ("P1", p.V1, ρ1))
    rmin = 0.1 * 2sqrt(3)                       # r = rmin / (2√3) = 0.1, as in filter_problem
    zs, xs, θs = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ,
                                     map = HelmholtzFilter(p.dΩ; rmin, ls = LUSolver()))
    zg, xg, θg = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ, map = HelmholtzFilter(p.Sf, p.dΩ))
    @test maximum(abs, get_free_dof_values(θs.ρ) .- get_free_dof_values(θg.ρ)) < 1e-12
    dΩ = p.dΩ
    yh = interpolate(cfun, p.Uf)
    gs = design_gradient!(zeros(length(zs)), θs, (ρ = δ -> ∫(yh * δ)dΩ,), nothing)
    gg = design_gradient!(zeros(length(zg)), θg, (ρ = δ -> ∫(yh * δ)dΩ,), nothing)
    @test maximum(abs, gs .- gg) < 1e-12
  end

  @testset "Short constructor of the filter: options" begin
    @test_throws UndefKeywordError HelmholtzFilter(p.dΩ)
    @test_throws r"give the measure of the design domain" HelmholtzFilter(; rmin = 1.0, ls = LUSolver())   # measure forgotten
    @test_throws r"give the measure of the design domain" CellMean()
    @test_throws ArgumentError HelmholtzFilter(p.dΩ; rmin = 0.0)
    @test_throws ArgumentError HelmholtzFilter(p.dΩ; rmin = 0.1, order = 0)
    # the solve given by the user is the one used
    n = Ref(0)
    F = HelmholtzFilter(p.dΩ; rmin = 0.2, ls = LUSolver(), solve = (m, b, ctx, stale) -> (n[] += 1; solve!(m, b; Assembly = stale)))
    z, x, θ = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0, map = F)
    @test n[] >= 1
    # order 2 and a filter on part of the domain
    F2 = HelmholtzFilter(p.dΩ; rmin = 0.2, order = 2, ls = LUSolver())
    z2, x2, θ2 = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0, map = F2)
    @test sum(∫(θ2.ρ)p.dΩ) ≈ sum(∫(x2.ρ)p.dΩ) rtol = 1e-10        # Neumann filter conserves the integral
    cell_ids = findall(i -> i <= num_cells(p.model) ÷ 2, 1:num_cells(p.model))
    Ωh = Triangulation(p.model, cell_ids)
    dΩh = Measure(Ωh, 2)
    Vh = TestFESpace(Ωh, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
    zh, xh, θh = single_field_design(:ρ, Vh; bounds = (0.0, 1.0), init = 0.5,
                                     map = HelmholtzFilter(dΩh; rmin = 0.2, ls = LUSolver()))
    @test maximum(abs, get_free_dof_values(θh.ρ) .- 0.5) < 1e-12      # constant field is kept
    # the backend of the filter model is set with model_backend (backend is the one of the vectors)
    zc, xc, θc = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0,
                                     map = HelmholtzFilter(p.dΩ; rmin = 0.2, ls = LUSolver(), model_backend = :compiled))
    zl, xl, θl = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0,
                                     map = HelmholtzFilter(p.dΩ; rmin = 0.2, ls = LUSolver(), backend = :lazy))
    @test maximum(abs, get_free_dof_values(θc.ρ) .- get_free_dof_values(θl.ρ)) < 1e-12
    # checks of the positional constructors and a hint for the old keyword
    @test_throws ArgumentError HelmholtzFilter(p.Sf, p.dΩ, :gpu)
    @test_throws ArgumentError CellMean(p.dΩ, :gpu)
    @test_throws MethodError CellMean(p.Ω)
  end

  @testset "CellMean" begin
    # P0 projection of the filtered field (the trick used before CellMean)
    F = HelmholtzFilter(p.Sf, p.dΩ)
    z, x, θ = single_field_design(:ρ, p.V1; bounds = (0.0, 1.0), init = ρ1, map = CellMean(p.dΩ) ∘ F)
    zf, xf, θf = single_field_design(:ρ, p.V1; bounds = (0.0, 1.0), init = ρ1, map = F)
    dΩ = p.dΩ
    ρf = θf.ρ
    ref = solve(LUSolver(), AffineFEOperator((u, v) -> ∫(u * v)dΩ, v -> ∫(ρf * v)dΩ, TrialFESpace(p.V0), p.V0))
    @test maximum(abs, get_free_dof_values(θ.ρ) .- get_free_dof_values(ref)) < 1e-12
    @test sum(∫(θ.ρ)dΩ) ≈ sum(∫(θf.ρ)dΩ) rtol = 1e-12             # the volume is kept
    # transpose: J = ∫(y CellMean(F(z))) is linear in z
    yh = interpolate(cfun, p.Uf)
    g = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(yh * δ)dΩ,), nothing)
    @test dot(g, get_values(z)) ≈ sum(∫(yh * θ.ρ)dΩ) rtol = 1e-12
    # Taylor test in a chain with a projection after the mean
    for backend in (:compiled, :lazy)
      H = TanhProjection(β = 4.0, η = 0.5)
      zt, xt, θt = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0,
                                       map = H ∘ CellMean(p.dΩ; backend) ∘ F)
      c = CellField(cfun, p.Ω)
      J() = (update_fields!(θt, nothing); sum(∫(c * θt.ρ * θt.ρ)dΩ))
      gt = design_gradient!(zeros(length(zt)), θt, (ρ = δ -> ∫(2 * c * θt.ρ * δ)dΩ,), nothing)
      rates = taylor_rates(J, gt, zt, [sin(0.7i) for i in 1:length(zt)])
      @test all(r -> 1.9 < r < 2.1, rates)
    end
    @test_throws ArgumentError CellMean(p.dΩ; backend = :gpu)
    # a volume after the mean stays linear
    zv, xv, θv = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0, map = CellMean(p.dΩ) ∘ F)
    @test HyperShape.Variables.is_linear_field(θv, :ρ)
  end

  @testset "PointwiseMap and Parameter" begin
    prm = Parameter(4.0)
    @test prm[] == 4.0 && state_version(prm) == 0
    prm[] = 4.0
    @test state_version(prm) == 0                   # same value: no change
    prm[] = 5.0
    @test prm[] == 5.0 && state_version(prm) == 1
    @test_throws ArgumentError Parameter(NaN)
    @test_throws ArgumentError (prm[] = Inf)
    @test_throws ArgumentError PointwiseMap(identity, one; parameters = (1.0,))
    prm[] = 4.0
    # the tanh projection written by the user equals TanhProjection (values and gradients), also after a change
    η = 0.5
    f(ρ) = (tanh(prm[] * η) + tanh(prm[] * (ρ - η))) / (tanh(prm[] * η) + tanh(prm[] * (1 - η)))
    df(ρ) = prm[] * (1 - tanh(prm[] * (ρ - η))^2) / (tanh(prm[] * η) + tanh(prm[] * (1 - η)))
    H = TanhProjection(β = 4.0, η = η)
    dΩ = p.dΩ
    c = CellField(cfun, p.Ω)
    for (name, V, ρ) in (("P0", p.V0, ρ0), ("P1", p.V1, ρ1))
      zp, xp, θp = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ,
                                       map = PointwiseMap(f, df; parameters = (prm,)) ∘ HelmholtzFilter(p.Sf, p.dΩ))
      zh, xh, θh = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ, map = H ∘ HelmholtzFilter(p.Sf, p.dΩ))
      for β in (4.0, 9.0)
        prm[] = β
        set_beta!(H, β)
        update_fields!(θp, nothing); update_fields!(θh, nothing)
        @test sum(∫(c * θp.ρ)dΩ) ≈ sum(∫(c * θh.ρ)dΩ) rtol = 1e-12
        gp = design_gradient!(zeros(length(zp)), θp, (ρ = δ -> ∫(c * δ)dΩ,), nothing)
        gh = design_gradient!(zeros(length(zh)), θh, (ρ = δ -> ∫(c * δ)dΩ,), nothing)
        @test maximum(abs, gp .- gh) < 1e-12 * maximum(abs, gh)
      end
      v = state_version(θp)
      prm[] = 2.0
      update_fields!(θp, nothing)
      @test state_version(θp) == v + 1               # a parameter of the map counts as a change
      prm[] = 4.0
      J() = (update_fields!(θp, nothing); sum(∫(c * θp.ρ * θp.ρ)dΩ))
      g = design_gradient!(zeros(length(zp)), θp, (ρ = δ -> ∫(2 * c * θp.ρ * δ)dΩ,), nothing)
      @test all(r -> 1.9 < r < 2.1, taylor_rates(J, g, zp, [sin(0.7i) for i in 1:length(zp)]))
    end
    @test !HyperShape.Variables.is_linear_field(PhysicalFields(DesignFields(OptVariables(ρ = FieldVariable(p.V0;
            bounds = (0.0, 1.0), init = 0.5)); ρ = Identity()); ρ = PointwiseMap(f, df)), :ρ)
    @test occursin("PointwiseMap", sprint(show, PointwiseMap(f, df)))
    @test PointwiseMap(f, df; parameters = prm).parameters === (prm,)        # a single parameter
    # a Parameter read by the form of a filter, declared as a dependency of its operator: a change re-solves it
    r = Parameter(0.1)
    dΩ = p.dΩ
    afr(u, v) = ∫(r[]^2 * ∇(u) ⋅ ∇(v) + u * v)dΩ
    Sr = LinearOperator(StaticLinearModel(afr, p.Uf, p.Vf, NothingBC(); ls = LUSolver()), (r,))
    zr, xr, θr = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0, map = HelmholtzFilter(Sr, p.dΩ))
    xh = FEFunction(p.V0, ρ0)
    for rv in (0.1, 0.3)
      r[] = rv
      v0 = state_version(θr)
      update_fields!(θr, nothing)
      rv == 0.3 && @test state_version(θr) == v0 + 1
      ref = solve(LUSolver(), AffineFEOperator((u, v) -> ∫(rv^2 * ∇(u) ⋅ ∇(v) + u * v)dΩ, v -> ∫(xh * v)dΩ, p.Uf, p.Vf))
      @test maximum(abs, get_free_dof_values(θr.ρ) .- get_free_dof_values(ref)) < 1e-12
    end
  end

  @testset "Exported names do not clash with Gridap and HyperFEM" begin
    M = Module()
    Core.eval(M, :(using Gridap, HyperFEM, HyperShape))
    for name in (:FieldVariable, :OptVariables, :DesignFields, :PhysicalFields, :Identity, :HelmholtzFilter,
                 :TanhProjection, :CellMean, :PointwiseMap, :Parameter, :single_field_design, :LinearOperator)
      @test Core.eval(M, name) === getfield(HyperShape, name)
    end
  end


  @testset "Frozen zones: z, x and θ (P$order, $backend)" for order in (0, 1), backend in (:compiled, :lazy)
    n = 8
    model = CartesianDiscreteModel((0, 1, 0, 1), (n, n))
    labels = get_face_labeling(model)
    add_tag_from_vertex_filter!(labels, "solid", model, x -> x[1] <= 0.25 + 1e-9 && x[2] >= 0.75 - 1e-9)   # 2×2 cells
    add_tag_from_vertex_filter!(labels, "void", model, x -> x[1] >= 0.75 - 1e-9 && x[2] <= 0.25 + 1e-9)    # 2×2 cells
    add_tag_from_vertex_filter!(labels, "edge", model, x -> x[2] ≈ 0)                                      # a line of nodes
    add_tag_from_vertex_filter!(labels, "next", model, x -> 0.25 - 1e-9 <= x[1] <= 0.5 + 1e-9 && x[2] >= 0.75 - 1e-9) # touches "solid"
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2)
    reffe = ReferenceFE(lagrangian, Float64, order)
    conf = order == 0 ? :L2 : :H1
    V = TestFESpace(model, reffe; conformity = conf, dirichlet_tags = ["solid", "void"])
    Vf = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    Sf = LinearOperator(StaticLinearModel((u, v) -> ∫(0.005 * ∇(u) ⋅ ∇(v) + u * v)dΩ, TrialFESpace(Vf), Vf, NothingBC();
                                          ls = LUSolver()); depends = ())
    H = TanhProjection(β = 4.0, η = 0.5)
    nfree = num_free_dofs(V)
    init = [0.3 + 0.4 * abs(sin(1.7i)) for i in 1:nfree]
    z, x, θ = single_field_design(:ρ, V; bounds = (0.0, 1.0), init, frozen = [1.0, 0.0], backend,
                                  map = H ∘ HelmholtzFilter(Sf, dΩ; backend))
    @test length(z) == nfree == num_free_dofs(TestFESpace(model, reffe; conformity = conf)) - (order == 0 ? 8 : 18)
    χ = frozen_indicator(x, :ρ)
    @test sum(∫(χ)dΩ) ≈ 8 / n^2
    zone = z.blocks[1].zone
    @test zone.ncells == 8 && zone.ndofs == (order == 0 ? 8 : 18)
    solid = CellField(x -> (x[1] < 0.25 && x[2] > 0.75) ? 1.0 : 0.0, Ω)
    void = CellField(x -> (x[1] > 0.75 && x[2] < 0.25) ? 1.0 : 0.0, Ω)
    err(f) = sum(∫(solid * (f - 1.0) * (f - 1.0) + void * f * f)dΩ)
    @test err(x.ρ) < 1e-28 && err(θ.ρ) == 0           # exact: the image of 1 and 0 by the projection
    # after a change of z and of β, still exact; the derivative vanishes in the zone
    set_values!(z, [0.2 + 0.6 * abs(cos(0.9i)) for i in 1:nfree])
    set_beta!(H, 8.0)
    update_fields!(θ, nothing)
    @test err(θ.ρ) == 0
    c = CellField(x -> 1 + x[1] * x[2], Ω)
    J() = (update_fields!(θ, nothing); sum(∫(c * θ.ρ * θ.ρ)dΩ))
    g = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(2 * c * θ.ρ * δ)dΩ,), nothing)
    @test all(r -> 1.9 < r < 2.1, taylor_rates(J, g, z, [sin(0.7i) for i in 1:length(z)]))
    gz = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫((solid + void) * δ)dΩ,), nothing)
    @test maximum(abs, gz) <= 1e-14                    # θ does not move in the zones
    # a linear chain stays linear (the mask is affine): the volume of the filtered density is a linear constraint
    z2, x2, θ2 = single_field_design(:ρ, V; bounds = (0.0, 1.0), init, frozen = [1.0, 0.0], backend,
                                     map = HelmholtzFilter(Sf, dΩ; backend))
    @test HyperShape.Variables.is_linear_field(θ2, :ρ)
    @test sum(∫(solid * (θ2.ρ - 1.0) * (θ2.ρ - 1.0) + void * θ2.ρ * θ2.ρ)dΩ) == 0
    # a chain of pointwise maps (as reaction–diffusion): φ ∈ [−1, 1] frozen at +1 and −1, θ = (1 + φ)/2
    hstep = PointwiseMap(φ -> (1 + φ) / 2, φ -> 0.5)
    z3, x3, θ3 = single_field_design(:φ, V; bounds = (-1.0, 1.0), init = 0.0, frozen = [1.0, -1.0], map = hstep, backend)
    @test sum(∫(solid * (θ3.φ - 1.0) * (θ3.φ - 1.0) + void * θ3.φ * θ3.φ)dΩ) == 0
    # errors when building
    Vplain = TestFESpace(model, reffe; conformity = conf)
    @test_throws ArgumentError FieldVariable(Vplain; bounds = (0.0, 1.0), init = 0.5, frozen = 1.0)
    @test_throws ArgumentError FieldVariable(V; bounds = (0.0, 1.0), init = 0.5)                     # no values
    @test_throws ArgumentError FieldVariable(V; bounds = (0.0, 1.0), init = 0.5, frozen = 2.0)       # outside the bounds
    @test_throws ArgumentError FieldVariable(TrialFESpace(V, 1.0); bounds = (0.0, 1.0), init = 0.5, frozen = 1.0)
    if order == 1                                       # a line of nodes is not a zone of cells
      Ve = TestFESpace(model, reffe; conformity = conf, dirichlet_tags = "edge")
      @test_throws ArgumentError FieldVariable(Ve; bounds = (0.0, 1.0), init = 0.5, frozen = 1.0)
    else                                                # in P0 a tag of a boundary freezes no cell: no Dirichlet value
      Ve = TestFESpace(model, reffe; conformity = conf, dirichlet_tags = "edge")
      @test num_dirichlet_dofs(Ve) == 0
    end
    @test_throws ArgumentError frozen_indicator(DesignFields(OptVariables(ρ = FieldVariable(Vplain; bounds = (0.0, 1.0), init = 0.5)); ρ = Identity()), :ρ)
    @test occursin("8 frozen cells", sprint(show, z.blocks[1])) && occursin("frozen zone", sprint(show, θ))
    @test_throws ArgumentError FieldVariable(V; bounds = (0.0, 1.0), init = 0.5, frozen = [1.0, 0.0, 0.5])  # 3 values, 2 tags
    # a value that is not dyadic stays exact (one value per cell); a value that follows a Parameter
    p = Parameter(2.0)
    pw = PointwiseMap(r -> r^p[], r -> p[] * r^(p[] - 1); parameters = (p,))
    z4, x4, θ4 = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = 0.5, frozen = [0.7, 0.3], map = pw ∘ HelmholtzFilter(Sf, dΩ; backend), backend)
    zerr(f, a, b) = sum(∫(solid * (f - a) * (f - a) + void * (f - b) * (f - b))dΩ)
    @test zerr(θ4.ρ, 0.7^2, 0.3^2) == 0
    p[] = 3.0
    update_fields!(θ4, nothing)
    @test zerr(θ4.ρ, 0.7^3, 0.3^3) == 0
    if order == 1
      # a function of x must be constant on each frozen cell; zones with different values must not touch
      @test_throws ArgumentError FieldVariable(V; bounds = (0.0, 1.0), init = 0.5, frozen = x -> 0.2 + 0.5x[1])
      Vt = TestFESpace(model, reffe; conformity = conf, dirichlet_tags = ["solid", "next"])
      @test_throws ArgumentError FieldVariable(Vt; bounds = (0.0, 1.0), init = 0.5, frozen = [1.0, 0.0])
    end
  end

  @testset "tag_adjacent_cells!: a boundary tag into a zone of cells (2D and 3D)" begin
    for (dim, n) in ((2, 6), (3, 4))
      model = CartesianDiscreteModel(dim == 2 ? (0, 1, 0, 1) : (0, 1, 0, 1, 0, 1), ntuple(_ -> n, dim))
      labels = get_face_labeling(model)
      top(x) = x[dim] ≈ 1
      add_tag_from_vertex_filter!(labels, "face", model, top)                                     # a face
      add_tag_from_vertex_filter!(labels, "edge", model, x -> top(x) && x[1] ≈ 0)                 # an edge (a corner in 2D)
      add_tag_from_vertex_filter!(labels, "corner", model, x -> top(x) && all(xi -> xi ≈ 0, x[1:dim-1]))   # a vertex
      tag_adjacent_cells!(labels, "face_zone", model, "face")
      tag_adjacent_cells!(labels, "face_zone2", model, "face"; layers = 2)
      tag_adjacent_cells!(labels, "edge_zone", model, "edge")
      tag_adjacent_cells!(labels, "corner_zone", model, "corner"; layers = 2)
      ncells(t) = count(Gridap.Geometry.get_face_mask(labels, t, dim))
      @test ncells("face_zone") == n^(dim - 1) && ncells("face_zone2") == 2n^(dim - 1)
      @test ncells("edge_zone") == (dim == 2 ? 1 : n) && ncells("corner_zone") == 2^dim
      for order in (0, 1), t in ("face_zone", "edge_zone", "corner_zone")
        V = TestFESpace(model, ReferenceFE(lagrangian, Float64, order); conformity = order == 0 ? :L2 : :H1, dirichlet_tags = t)
        fv = FieldVariable(V; bounds = (0.0, 1.0), init = 0.5, frozen = 1.0)
        @test fv.zone.ncells == ncells(t)
      end
      # the original tags are kept
      @test count(Gridap.Geometry.get_face_mask(labels, "face", dim - 1)) == n^(dim - 1)
      @test_throws ArgumentError tag_adjacent_cells!(labels, "face_zone", model, "face")      # exists
      @test_throws ArgumentError tag_adjacent_cells!(labels, "x", model, "face"; layers = 0)
      @test_throws ArgumentError tag_adjacent_cells!(labels, "x", model, "nope")
      # a tag with cells starts from its cells: one more layer each time
      tag_adjacent_cells!(labels, "face_zone3", model, "face_zone2")
      @test ncells("face_zone3") == 2n^(dim - 1)
      tag_adjacent_cells!(labels, "face_zone4", model, "face_zone2"; layers = 2)
      @test ncells("face_zone4") == 3n^(dim - 1)
    end
  end

  @testset "Frozen zones with a periodic design space" begin
    model = CartesianDiscreteModel((0, 1, 0, 1), (8, 8); isperiodic = (true, false))
    labels = get_face_labeling(model)
    add_tag_from_vertex_filter!(labels, "solid", model, x -> x[2] >= 0.75 - 1e-9)
    Ω = Triangulation(model)
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1, dirichlet_tags = "solid")
    z, x, θ = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = 0.5, frozen = 1.0, map = Identity())
    @test z.blocks[1].zone.ncells == 16
    @test sum(∫(frozen_indicator(x, :ρ) * (θ.ρ - 1.0) * (θ.ρ - 1.0))Measure(Ω, 2)) == 0
  end

  include("interface.jl")
  include("hole_seed.jl")

end
