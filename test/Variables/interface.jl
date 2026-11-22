# Interface maps Step() and Ersatz(), design_measures (included by test/Variables/runtests.jl)
using HyperShape.Variables: cell_fraction, quad_fraction, hex_fraction, is_differentiable_field
using Random

# area of the part of a triangle where the linear interpolant of φ is ≥ 0, by clipping the polygon (independent of
# the closed forms of cell_fraction)
function _clip_area(P, φ)
  poly, vals = collect(P), collect(φ)
  out = Tuple{Float64,Float64}[]
  for k in eachindex(poly)
    a, b = poly[k], poly[mod1(k + 1, length(poly))]
    fa, fb = vals[k], vals[mod1(k + 1, length(poly))]
    fa >= 0 && push!(out, a)
    if (fa >= 0) != (fb >= 0)
      t = fa / (fa - fb)
      push!(out, (a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])))
    end
  end
  length(out) < 3 && return 0.0
  abs(sum(out[k][1] * out[mod1(k + 1, length(out))][2] - out[mod1(k + 1, length(out))][1] * out[k][2]
          for k in eachindex(out))) / 2
end

# heat-like problem on the unit square/cube with φ ∈ [−1, 1] through `map` (the prototype of the design adviser)
function interface_problem(; n = 8, dim = 2, simplex = false, map = Ersatz(), degree = 2, backend = :compiled,
                           init = x -> x[1] - 0.37, filter = false, maxfrac = 0.5, frozen = false)
  model = CartesianDiscreteModel(dim == 2 ? (0, 1, 0, 1) : (0, 1, 0, 1, 0, 1), ntuple(_ -> n, dim))
  simplex && (model = simplexify(model))
  labels = get_face_labeling(model)
  add_tag_from_vertex_filter!(labels, "left", model, x -> x[1] < 1e-9)
  add_tag_from_vertex_filter!(labels, "solid", model, x -> x[1] >= 0.75 - 1e-9 && x[2] >= 0.75 - 1e-9)
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  reffe = ReferenceFE(lagrangian, Float64, 1)
  Vφ = frozen ? TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = "solid") : TestFESpace(model, reffe; conformity = :H1)
  z = OptVariables(φ = FieldVariable(Vφ; bounds = (-1.0, 1.0), init = x -> clamp(init(x), -1.0, 1.0),
                                    frozen = frozen ? 1.0 : nothing))
  x = DesignFields(z; φ = Identity())
  m = filter ? map ∘ HelmholtzFilter(dΩ; rmin = 1.0 / n, ls = LUSolver()) : map
  θ = PhysicalFields(x; ρ = :φ => m)
  dΩd = design_measures(θ; degree)
  bc = BoundaryConditions(BoundaryCondition("left", 0.0))
  V = TestFESpace(Ω, reffe, bc; conformity = :H1)
  U = TrialFESpace(V, bc, 1.0)
  k(χ) = 1e-3 + (1 - 1e-3) * χ
  dk(χ) = 1 - 1e-3
  src(x) = 1 + x[1]
  a(u, v) = ∫((k ∘ θ.ρ) * ∇(u) ⋅ ∇(v))dΩd.domain
  l(v) = ∫(src * v)dΩ
  S = StateOperator(StaticLinearModel(l, a, U, V, bc; ls = LUSolver(), backend), θ;
        adjoint = AdjointProblem(sensitivity = (u, p) -> (δ -> ∫((dk ∘ θ.ρ) * ∇(u) ⋅ ∇(p) * δ)dΩd.domain)))
  J = Functional((u, θ) -> ∫(src * u)dΩ, S; dJdu = SelfAdjoint(), scale = 1.0)
  Vol = Volume(θ, dΩd; max_fraction = maxfrac)
  (; model, Ω, dΩ, dΩd, Vφ, z, x, θ, S, J, Vol)
end

@testset "Interface maps: Step(), Ersatz(), design_measures" begin
  rng = Xoshiro(7)

  @testset "Kernels of Ersatz: exact fractions and derivatives" begin
    T = ((0.0, 0.0), (1.0, 0.0), (0.3, 0.8))
    area = 0.5 * abs(1.0 * 0.8 - 0.0 * 0.3)
    for _ in 1:2000
      φ = Tuple(2 .* rand(rng, 3) .- 1)
      f, g = cell_fraction(φ)
      @test f ≈ _clip_area(T, φ) / area atol = 1e-13
      # derivative against centred differences
      h = 1e-6
      for i in 1:3
        φp, φm = Base.setindex(φ, φ[i] + h, i), Base.setindex(φ, φ[i] - h, i)
        (minimum(abs, φ) > 1e-3) && @test g[i] ≈ (cell_fraction(φp)[1] - cell_fraction(φm)[1]) / 2h atol = 1e-6
      end
      @test abs(sum(φ .* g)) <= 1e-10 * (1 + maximum(abs, g))          # homogeneous of degree 0
      @test cell_fraction(.-φ)[1] ≈ 1 - f atol = 1e-14                    # complement
    end
    for _ in 1:2000
      φ = Tuple(2 .* rand(rng, 4) .- 1)
      f, g = cell_fraction(φ)
      @test 0 <= f <= 1
      @test cell_fraction(.-φ)[1] ≈ 1 - f atol = 1e-13
      @test cell_fraction((φ[3], φ[1], φ[4], φ[2]))[1] ≈ f atol = 1e-14  # permutations
      h = 1e-6
      for i in 1:4
        φp, φm = Base.setindex(φ, φ[i] + h, i), Base.setindex(φ, φ[i] - h, i)
        (minimum(abs, φ) > 1e-3) && @test g[i] ≈ (cell_fraction(φp)[1] - cell_fraction(φm)[1]) / 2h atol = 1e-6
      end
      @test abs(sum(φ .* g)) <= 1e-10 * (1 + maximum(abs, g))
    end
    # nearly equal values: no cancellation
    ε = 1e-12
    @test cell_fraction((1.0 + ε, -1.0, -1.0 - ε))[1] ≈ 0.25 rtol = 1e-10
    @test cell_fraction((1.0, 1.0 + ε, -1.0, -1.0 - ε))[1] ≈ 0.5 rtol = 1e-10
    # interface through a vertex or along an edge
    @test cell_fraction((0.0, -1.0, -1.0)) == (0.0, (0.0, 0.0, 0.0)) && cell_fraction((0.0, 0.0, 0.0))[1] == 1.0
    @test cell_fraction((0.0, 0.0, -1.0, -1.0))[1] == 0.0
    # quadrilateral and hexahedron with φ linear: exact (φ = x + y − 0.25 on the unit square: 1 − 1/32)
    w4, w24 = (0.25, 0.25, 0.25, 0.25), ntuple(_ -> 1 / 24, 24)
    @test quad_fraction((-0.25, 0.75, 0.75, 1.75), w4)[1] ≈ 1 - 1 / 32 atol = 1e-14
    hv = Tuple(x + y + z - 0.5 for z in (0, 1) for y in (0, 1) for x in (0, 1))
    @test hex_fraction(hv, w24)[1] ≈ 1 - 0.5^3 / 6 atol = 1e-14
    # symmetric subdivision: a reflected field gives the same fraction
    for _ in 1:200
      q = Tuple(2 .* rand(rng, 4) .- 1)
      @test quad_fraction((q[2], q[1], q[4], q[3]), w4)[1] ≈ quad_fraction(q, w4)[1] atol = 1e-14
      hq = Tuple(2 .* rand(rng, 8) .- 1)
      @test hex_fraction((hq[2], hq[1], hq[4], hq[3], hq[6], hq[5], hq[8], hq[7]), w24)[1] ≈ hex_fraction(hq, w24)[1] atol = 1e-13
      # derivatives of the subdivisions
      f, g = quad_fraction(q, w4)
      h = 1e-6
      if minimum(abs, q) > 1e-3 && abs(sum(q) / 4) > 1e-3
        for i in 1:4
          @test g[i] ≈ (quad_fraction(Base.setindex(q, q[i] + h, i), w4)[1] - quad_fraction(Base.setindex(q, q[i] - h, i), w4)[1]) / 2h atol = 1e-5
        end
      end
    end
    # no allocations
    φ3, φ4, φ8 = (0.3, -0.2, 0.1), (0.3, -0.2, 0.1, -0.4), Tuple(2 .* rand(rng, 8) .- 1)
    cell_fraction(φ3); cell_fraction(φ4); quad_fraction(φ4, w4); hex_fraction(φ8, w24)
    @test @allocated(cell_fraction(φ3)) == 0 && @allocated(cell_fraction(φ4)) == 0
    @test @allocated(quad_fraction(φ4, w4)) == 0 && @allocated(hex_fraction(φ8, w24)) == 0
  end

  @testset "Ersatz: exact volume of a half space, consistency on the four cells, with a filter and a frozen zone" begin
    for (dim, simplex) in ((2, true), (2, false), (3, true), (3, false))
      n = dim == 2 ? 8 : 4
      # an inclined half space: the volume is exact (with any degree)
      c = (0.31, 0.27, 0.22)
      f(x) = sum(c[i] * (x[i] - 0.5) for i in 1:dim) + 0.05
      for degree in (1, 2, 6)
        p = interface_problem(; n, dim, simplex, degree, init = f)
        vol = sum(∫(p.θ.ρ)p.dΩd.domain)
        # exact: the volume of {f ≥ 0} in the unit cube, by Monte Carlo would be loose; f is linear, so the
        # subdivision is exact: compare with the fraction of a much finer mesh
        q = interface_problem(; n = 2n, dim, simplex, degree, init = f)
        @test vol ≈ sum(∫(q.θ.ρ)q.dΩd.domain) atol = 1e-13
      end
      p = interface_problem(; n, dim, simplex)
      rep = check_consistency((J = p.J, volume = p.Vol))
      @test rep.passed
      p = interface_problem(; n, dim, simplex, filter = true)
      @test check_consistency((J = p.J,)).passed
    end
    p = interface_problem(; frozen = true, init = x -> x[1] + x[2] - 1.2)
    @test check_consistency((J = p.J,)).passed
    @test p.z.blocks[1].zone !== nothing
    # the material of a frozen zone φ = 1 is 1
    χ = get_free_dof_values(p.θ.ρ isa Gridap.FESpaces.FEFunction ? p.θ.ρ : interpolate(p.θ.ρ, FESpace(p.Ω, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)))
    @test maximum(χ) == 1.0
    # :lazy and :compiled give the same material and gradient
    pc, pl = interface_problem(; backend = :compiled, map = Ersatz(backend = :compiled)), interface_problem(; backend = :lazy, map = Ersatz(backend = :lazy))
    gc = functional_gradient!(zeros(length(pc.z)), pc.J, OptContext(:check))
    gl = functional_gradient!(zeros(length(pl.z)), pl.J, OptContext(:check))
    @test gc ≈ gl rtol = 1e-10
    # without interface the derivative vanishes: check_consistency says why
    p = interface_problem(; init = x -> 0.5)
    rep = check_consistency((J = p.J,))
    @test any(r -> occursin("no cell is cut", r.message), rep.results)
  end

  @testset "Step: values, no derivative, the error of DensityMethod, not applicable in check_consistency" begin
    p = interface_problem(; map = Step(), degree = 4)
    @test !is_differentiable_field(p.θ, :ρ)
    # the step at the quadrature points: φ_h = x − 0.37 (exact in P1)
    vals = Gridap.CellData.evaluate(p.θ.ρ, get_cell_points(p.dΩd.domain.quad))
    @test all(v -> v == 0.0 || v == 1.0, Iterators.flatten(collect(vals)))
    @test sum(∫(p.θ.ρ)p.dΩd.domain) ≈ 0.63 atol = 0.06
    pl = interface_problem(; map = Step(), degree = 4, backend = :lazy)
    @test sum(∫(p.θ.ρ)p.dΩd.domain) == sum(∫(pl.θ.ρ)pl.dΩd.domain)
    rep = check_consistency((J = p.J,))
    @test rep.passed && any(r -> !r.applicable && occursin("Step()", r.message), rep.results)
    @test occursin("not applicable", sprint(show, MIME"text/plain"(), rep))
    problem = Optimization(p.z; objective = p.J, constraints = (volume = p.Vol,))
    @test_throws r"Step\(\)" solve!(problem, DensityMethod(SiMPL(); metric = LumpedMass()); maxiter = 2, verbose = 0)
    @test_throws ArgumentError functional_gradient!(zeros(length(p.z)), p.J, OptContext(:check))
    # a frozen zone through Step(): φ = 1 gives 1
    pf = interface_problem(; map = Step(), frozen = true, init = x -> -0.5)
    @test sum(∫(pf.θ.ρ)pf.dΩd.domain) >= 1 / 16 - 1e-12
  end

  @testset "design_measures and errors of construction" begin
    p = interface_problem()
    d = design_measures(p.θ; degree = 3)
    @test d.domain isa Measure && propertynames(d) == (:domain,)
    @test_throws r"reserved" d.solid
    @test_throws ArgumentError d.foo
    @test_throws ArgumentError design_measures(p.θ; degree = 3, field = :φ)
    # a design over part of the domain: the measure of its triangulation
    model = CartesianDiscreteModel((0, 1, 0, 1), (4, 4))
    Ωd = Triangulation(model, findall(c -> c <= 8, 1:16))
    Vd = TestFESpace(Ωd, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    zd = OptVariables(φ = FieldVariable(Vd; bounds = (-1.0, 1.0), init = x -> x[2] - 0.2))
    θd = PhysicalFields(DesignFields(zd; φ = Identity()); ρ = :φ => Ersatz())
    @test sum(∫(1.0)design_measures(θd; degree = 1).domain) ≈ 0.5
    # Ersatz needs a nodal P1 field
    V0 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
    V2 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 2); conformity = :H1)
    V1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    for V in (V0, V2)
      x = DesignFields(OptVariables(φ = FieldVariable(V; bounds = (-1.0, 1.0), init = 0.1)); φ = Identity())
      @test_throws r"nodal P1" PhysicalFields(x; ρ = :φ => Ersatz())
    end
    x = DesignFields(OptVariables(φ = FieldVariable(V1; bounds = (-1.0, 1.0), init = 0.1)); φ = Identity())
    @test_throws r"nodal P1" PhysicalFields(x; ρ = :φ => Ersatz() ∘ TanhProjection(β = 2.0, η = 0.5))
    # init as a function of the coordinates
    zf = OptVariables(φ = FieldVariable(V1; bounds = (-1.0, 1.0), init = x -> x[1] - 0.5))
    @test get_values(zf) ≈ get_free_dof_values(interpolate(x -> x[1] - 0.5, V1))
  end

  @testset "Ersatz in the loop of DensityMethod (smoke test), an infeasible start, allocations" begin
    holes(x) = -0.3 + 0.8cos(4π * x[1]) * cos(4π * x[2])
    p = interface_problem(; n = 16, init = holes, filter = true, maxfrac = 0.4)
    problem = Optimization(p.z; objective = p.J, constraints = (volume = p.Vol,))
    r = solve!(problem, DensityMethod(SiMPL(); metric = LumpedMass()); maxiter = 15, verbose = 0, stop = nothing)
    @test r.history.objective[end] < r.history.objective[1]
    @test r.history.constraints.volume[end] <= 1e-3
    # an infeasible start whose constraint depends on entries without derivative there: every entry is scaled
    p = interface_problem(; n = 8, init = x -> x[1] - 0.2, maxfrac = 0.5)
    problem = Optimization(p.z; objective = p.J, constraints = (volume = p.Vol,))
    r = solve!(problem, DensityMethod(SiMPL(); metric = LumpedMass()); maxiter = 3, verbose = 0, stop = nothing)
    @test r.start_scaling !== nothing && r.history.constraints.volume[1] <= 1e-3
    # no allocations per cell: update and pullback of the stage allocate a fixed amount (Gridap's accessors)
    allocs = map((16, 32)) do n
      p = interface_problem(; n)
      s = last(getfield(p.θ, :chains)[1])
      HyperShape.Variables.update_stage!(s, nothing)
      a1 = @allocated(HyperShape.Variables.update_stage!(s, nothing))
      fill!(s.acc, 1.0); s.pending[] = true
      HyperShape.Variables.flush!(s, nothing)
      fill!(s.acc, 1.0); s.pending[] = true
      a2 = @allocated(HyperShape.Variables.flush!(s, nothing))
      (a1, a2)
    end
    @test allocs[1] == allocs[2] && all(<=(2048), allocs[1])
  end
end
