using Gridap
using Gridap.FESpaces
using HyperFEM
using HyperFEMKernels
using HyperShape
using HyperShape.Variables: update_fields!, design_gradient!, set_values!, get_values, get_bounds, changed!,
                            set_beta!, block_range, HeavisideValue, version
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
    v = version(z)
    set_values!(z, zeros(length(z)))
    @test version(z) == v + 1
    changed!(z)
    @test version(z) == v + 2
    @test_throws DimensionMismatch set_values!(z, zeros(3))
    @test_throws ArgumentError OptVariables()
    @test_throws ArgumentError OptVariables(a = p.V0)
    @test_throws ArgumentError FieldVariable(p.V0; bounds = (1.0, 1.0), init = 1.0)
    @test_throws ArgumentError FieldVariable(p.V0; bounds = (0.0, 1.0), init = 1.5)
    @test_throws DimensionMismatch FieldVariable(p.V0; bounds = (0.0, 1.0), init = [0.5])
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
    vx = version(x)
    update_fields!(x, nothing)
    @test version(x) == vx                         # nothing changed
    w = copy(get_values(z)); w[1] = 0.9
    set_values!(z, w)
    update_fields!(x, nothing)
    @test version(x) == vx + 1
    @test x.a === fa && get_free_dof_values(x.a)[1] == 0.9    # same object, updated in place
    @test_throws ArgumentError DesignFields(z; a = Identity())
    @test_throws ArgumentError DesignFields(z; a = Identity(), b = Identity(), c = Identity())
    @test_throws ArgumentError DesignFields(z; a = Identity(), b = Heaviside(β = 1.0, η = 0.5))
  end

  @testset "Helmholtz filter = direct solve ($(name))" for (name, V, ρ) in (("P0", p.V0, ρ0), ("P1", p.V1, ρ1))
    z, x, θ = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ,
                                  physical = HelmholtzFilter(p.Sf; measure = p.dΩ))
    xh = FEFunction(V, ρ)
    dΩ = p.dΩ
    ref = solve(LUSolver(), AffineFEOperator(p.af, v -> ∫(xh * v)dΩ, p.Uf, p.Vf))
    @test θ.ρ isa FEFunction
    @test maximum(abs, get_free_dof_values(θ.ρ) .- get_free_dof_values(ref)) < 1e-12
  end

  @testset "Transpose of the filter ($(name))" for (name, V, ρ) in (("P0", p.V0, ρ0), ("P1", p.V1, ρ1))
    # J(z) = ∫(y F(z)) dΩ is linear in z with J(0) = 0, so J(z) = g⋅z exactly
    z, x, θ = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ,
                                  physical = HelmholtzFilter(p.Sf; measure = p.dΩ))
    yh = interpolate(cfun, p.Uf)
    dΩ = p.dΩ
    g = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(yh * δ)dΩ,), nothing)
    J = sum(∫(yh * θ.ρ)dΩ)
    @test dot(g, get_values(z)) ≈ J rtol = 1e-12
  end

  @testset "Taylor test: projection ∘ filter ($(name))" for (name, V, ρ) in (("P0", p.V0, ρ0), ("P1", p.V1, ρ1))
    H = Heaviside(β = 4.0, η = 0.5)
    z, x, θ = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ,
                                  physical = H ∘ HelmholtzFilter(p.Sf; measure = p.dΩ))
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
    H = Heaviside(β = 6.0, η = 0.4)
    z = OptVariables(ρ = FieldVariable(p.V0; bounds = (0.0, 1.0), init = ρ0))
    x = DesignFields(z; ρ = Identity())
    θ = PhysicalFields(x; ρ = H ∘ HelmholtzFilter(p.Sf; measure = p.dΩ), ρf = :ρ => HelmholtzFilter(p.Sf; measure = p.dΩ))
    dΩ = p.dΩ
    hv = HeavisideValue(H)
    @test sum(∫(θ.ρ)dΩ) ≈ sum(∫(Operation(hv)(θ.ρf))dΩ) rtol = 1e-14
    @test !(sum(∫(θ.ρ)dΩ) ≈ sum(∫(Operation(hv)(x.ρ))dΩ))
  end

  @testset "Several physical fields from one design field; two design blocks" begin
    H = Heaviside(β = 3.0, η = 0.5)
    z = OptVariables(a = FieldVariable(p.V0; bounds = (0.0, 1.0), init = ρ0),
                     b = FieldVariable(p.V1; bounds = (0.0, 1.0), init = ρ1))
    x = DesignFields(z; a = Identity(), b = Identity())
    θ = PhysicalFields(x; ρ = :a => H ∘ HelmholtzFilter(p.Sf; measure = p.dΩ),
                          E = :a => HelmholtzFilter(p.Sf; measure = p.dΩ),
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
    @test_throws ArgumentError PhysicalFields(x; ρ = HelmholtzFilter(p.Sf; measure = p.dΩ))
    @test_throws ArgumentError PhysicalFields(x; ρ = :c => Identity(), b = Identity())
    @test_throws ArgumentError PhysicalFields(x; a = 1.0, b = Identity())
    @test_throws ArgumentError design_gradient!(zeros(length(z)), θ, (q = δ -> ∫(δ)dΩ,), nothing)
  end

  @testset "Versions and number of filter solves" begin
    H = Heaviside(β = 2.0, η = 0.5)
    z, x, θ = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0,
                                  physical = H ∘ HelmholtzFilter(p.Sf; measure = p.dΩ))
    n = p.nsolves[]
    vθ = version(θ)
    update_fields!(θ, nothing)
    @test p.nsolves[] == n && version(θ) == vθ          # nothing changed: no solve
    changed!(z)
    update_fields!(θ, nothing)
    @test p.nsolves[] == n + 1 && version(θ) == vθ + 1  # one filter solve
    set_beta!(H, 4.0)
    update_fields!(θ, nothing)
    @test p.nsolves[] == n + 1 && version(θ) == vθ + 2  # β counts as a change; the projection is lazy: no solve
    set_beta!(H, 4.0)
    update_fields!(θ, nothing)
    @test version(θ) == vθ + 2                           # same β: no change
    dΩ = p.dΩ
    g1 = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(δ)dΩ,), nothing)
    @test p.nsolves[] == n + 2                           # one transposed solve
    g2 = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(2 * δ)dΩ,), nothing)
    @test p.nsolves[] == n + 3 && g2 ≈ 2 .* g1           # another form, same design: one solve, no update
    # projection before the filter: a new β changes the input of the filter
    H2 = Heaviside(β = 2.0, η = 0.5)
    z2, x2, θ2 = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0,
                                     physical = HelmholtzFilter(p.Sf; measure = p.dΩ) ∘ H2)
    n = p.nsolves[]
    set_beta!(H2, 5.0)
    update_fields!(θ2, nothing)
    @test p.nsolves[] == n + 1
    @test_throws ArgumentError Heaviside(β = 0.0, η = 0.5)
    @test_throws ArgumentError Heaviside(β = 1.0, η = 1.0)
    @test_throws ArgumentError set_beta!(H, -1.0)
    @test_throws ArgumentError (H.η = 0.3)
    @test_throws ArgumentError (H.β = 3.0)
    @test_throws DimensionMismatch design_gradient!(zeros(3), θ, (ρ = δ -> ∫(δ)dΩ,), nothing)
    @test_throws ArgumentError θ.foo
    @test_throws ArgumentError x.foo
  end

  @testset "Values of z are checked" begin
    z, x, θ = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0, physical = Identity())
    v = version(z)
    w = copy(get_values(z)); w[3] = NaN
    @test_throws ArgumentError set_values!(z, w)
    w[3] = 1.5
    @test_throws ArgumentError set_values!(z, w)
    get_values(z)[2] = -0.1
    @test_throws ArgumentError changed!(z)
    @test version(z) == v
  end

  @testset "Filter with homogeneous Dirichlet values" begin
    Vd = TestFESpace(p.model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1, dirichlet_tags = "boundary")
    Ud = TrialFESpace(Vd, 0.0)
    af, dΩ = p.af, p.dΩ
    Sd = LinearOperator(StaticLinearModel(af, Ud, Vd, NothingBC(); ls = LUSolver()); depends = (),
                        solve = (m, b, ctx, stale) -> solve!(m, b; Assembly = stale))
    z, x, θ = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0,
                                  physical = Heaviside(β = 3.0, η = 0.5) ∘ HelmholtzFilter(Sd; measure = dΩ))
    xh = FEFunction(p.V0, ρ0)
    ref = solve(LUSolver(), AffineFEOperator(af, v -> ∫(xh * v)dΩ, Ud, Vd))
    filtered = PhysicalFields(x; ρ = HelmholtzFilter(Sd; measure = dΩ))
    @test maximum(abs, get_free_dof_values(filtered.ρ) .- get_free_dof_values(ref)) < 1e-12
    c = CellField(cfun, p.Ω)
    J() = (update_fields!(θ, nothing); sum(∫(c * θ.ρ * θ.ρ)dΩ))
    g = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(2 * c * θ.ρ * δ)dΩ,), nothing)
    @test all(r -> 1.9 < r < 2.1, taylor_rates(J, g, z, [sin(0.7i) for i in 1:length(z)]))
    # non-homogeneous values are rejected (the load vector carries no lift)
    Un = TrialFESpace(Vd, 1.0)
    Sn = LinearOperator(StaticLinearModel(af, Un, Vd, NothingBC(); ls = LUSolver()); depends = (),
                        solve = (m, b, ctx, stale) -> solve!(m, b; Assembly = stale))
    @test_throws ArgumentError PhysicalFields(x; ρ = HelmholtzFilter(Sn; measure = dΩ))
  end

  @testset "Transpose with independent vectors ($(name))" for (name, V, ρ) in (("P0", p.V0, ρ0), ("P1", p.V1, ρ1))
    # g is the gradient of z ↦ ∫(y F(z)) dΩ at one design; ∫(y F(z₂)) dΩ = g⋅z₂ for another design z₂
    z, x, θ = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ,
                                  physical = HelmholtzFilter(p.Sf; measure = p.dΩ))
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
    F = HelmholtzFilter(p.Sf; measure = p.dΩ)
    map = mk(F, Heaviside(β = 3.0, η = 0.45), Heaviside(β = 2.0, η = 0.55))
    z, x, θ = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ, physical = map)
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
    θ = PhysicalFields(x; a = Heaviside(β = 3.0, η = 0.5) ∘ HelmholtzFilter(p.Sf; measure = p.dΩ),
                          b = Heaviside(β = 5.0, η = 0.4) ∘ HelmholtzFilter(p.Sf; measure = p.dΩ))
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
                                    physical = Heaviside(β = 3.0, η = 0.5) ∘ HelmholtzFilter(q.Sf; measure = q.dΩ))
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
    @test_throws UndefKeywordError LinearOperator(op.model; solve = op.solve)
    bad = LinearOperator(op.model; depends = (), solve = (m, b, ctx, stale) -> zeros(3))
    @test_throws DimensionMismatch HyperShape.Operators.solve_linear!(y, bad, b, nothing)
  end

  @testset "Weak forms capture θ: lazy and compiled backends" begin
    H = Heaviside(β = 4.0, η = 0.5)
    z, x, θ = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0,
                                  physical = H ∘ HelmholtzFilter(p.Sf; measure = p.dΩ))
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
      H = Heaviside(β = 3.0, η = 0.45)
      z, x, θ = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ, backend,
                                    physical = mk(HelmholtzFilter(p.Sf; measure = dΩ, backend), H))
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
      H = Heaviside(β = 3.0, η = 0.45)
      z, x, θ = single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = ρ0, backend,
                                    physical = H ∘ HelmholtzFilter(p.Sf; measure = dΩ, backend))
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
    @test length(assems[1].ops) <= HyperShape.Variables.MAX_COMPILED_FORMS
    filter_stage = getfield(θc, :chains)[1][1]
    @test length(filter_stage.assem.ops) <= HyperShape.Variables.MAX_COMPILED_FORMS
    # every cached operator shares the tables of the first one
    @test all(op -> op.fields === assems[1].ops[1].fields && op.colors === assems[1].ops[1].colors, assems[1].ops)
  end

  @testset "Taylor test with the lazy backend ($(name))" for (name, V, ρ) in (("P0", p.V0, ρ0), ("P1", p.V1, ρ1))
    z, x, θ = single_field_design(:ρ, V; bounds = (0.0, 1.0), init = ρ, backend = :lazy,
                                  physical = Heaviside(β = 4.0, η = 0.5) ∘ HelmholtzFilter(p.Sf; measure = p.dΩ, backend = :lazy))
    c = CellField(cfun, p.Ω)
    dΩ = p.dΩ
    J() = (update_fields!(θ, nothing); sum(∫(c * θ.ρ * θ.ρ)dΩ))
    g = design_gradient!(zeros(length(z)), θ, (ρ = δ -> ∫(2 * c * θ.ρ * δ)dΩ,), nothing)
    @test all(r -> 1.9 < r < 2.1, taylor_rates(J, g, z, [sin(0.7i) for i in 1:length(z)]))
  end

  @testset "Backend options" begin
    zb = OptVariables(ρ = FieldVariable(p.V0; bounds = (0.0, 1.0), init = ρ0))
    @test_throws ArgumentError PhysicalFields(DesignFields(zb; ρ = Identity()); ρ = Identity(), backend = Identity())
    @test_throws ArgumentError HelmholtzFilter(p.Sf; measure = p.dΩ, backend = :gpu)
    z = OptVariables(ρ = FieldVariable(p.V0; bounds = (0.0, 1.0), init = ρ0))
    x = DesignFields(z; ρ = Identity())
    @test_throws ArgumentError PhysicalFields(x; ρ = Identity(), backend = :gpu)
    @test_throws ArgumentError single_field_design(:ρ, p.V0; bounds = (0.0, 1.0), init = 0.5, physical = Identity(), backend = :gpu)
    F = HelmholtzFilter(p.Sf; measure = p.dΩ)
    @test F.backend === :compiled                                    # the default
    @test HelmholtzFilter(p.Sf; measure = p.dΩ, backend = :lazy).backend === :lazy
  end

  @testset "Exported names do not clash with Gridap and HyperFEM" begin
    M = Module()
    Core.eval(M, :(using Gridap, HyperFEM, HyperShape))
    for name in (:FieldVariable, :OptVariables, :DesignFields, :PhysicalFields, :Identity, :HelmholtzFilter,
                 :Heaviside, :single_field_design, :LinearOperator)
      @test Core.eval(M, name) === getfield(HyperShape, name)
    end
  end

end
