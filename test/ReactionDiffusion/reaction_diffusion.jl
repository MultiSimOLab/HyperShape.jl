# ReactionDiffusion: the step, the operator, the exact multiplier of the volume, the normalisations, the :physical
# sensitivity, the loop (stop, rejection, checkpoint, augmented Lagrangian, preset) and the errors
using Test
using LinearAlgebra
using SparseArrays
using Gridap
using Gridap.FESpaces
using HyperFEM
using HyperShape
using HyperShape.Functionals: get_scale

const RDM = HyperShape.Methods
const OPT = HyperShape.Optimizers

# Cantilever 2 × 1: clamped on the left, unit traction downwards on the middle fifth of the right side, plane stress.
# design = :levelset (φ ∈ [−1, 1] through `interface`, E linear in χ) or :density (ρ ∈ [0, 1], SIMP p = 3).
function rd_cantilever(; nx = 40, ny = 20, design = :levelset, interface = Ersatz(), frozen = true, simplex = false,
                       fraction = 0.5, init = nothing, solve_flag = nothing, order = 1, bounds = nothing, left_cap = nothing)
  model = CartesianDiscreteModel((0, 2, 0, 1), (nx, ny))
  simplex && (model = simplexify(model))
  labels = get_face_labeling(model)
  add_tag_from_vertex_filter!(labels, "clamp", model, x -> x[1] < 1e-9)
  add_tag_from_vertex_filter!(labels, "load", model, x -> x[1] > 2 - 1e-9 && abs(x[2] - 0.5) <= 0.1 + 1e-9)
  frozen && tag_adjacent_cells!(labels, "load_zone", model, "load")
  Ω = Triangulation(model)
  dΓ = Measure(BoundaryTriangulation(model; tags = "load"), 2)
  bc = BoundaryConditions(BoundaryCondition("clamp", VectorValue(0.0, 0.0)))
  V = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1), bc; conformity = :H1)
  U = TrialFESpace(V, bc, 1.0)
  reffe = ReferenceFE(lagrangian, Float64, order)
  conf = order == 0 ? :L2 : :H1
  Vφ = frozen ? TestFESpace(model, reffe; conformity = conf, dirichlet_tags = "load_zone") : TestFESpace(model, reffe; conformity = conf)
  lb, φ0 = design === :levelset ? (-1.0, 1.0) : (0.0, fraction)
  init === nothing || (φ0 = init)
  bnds = bounds === nothing ? (lb, 1.0) : bounds
  z = OptVariables(φ = FieldVariable(Vφ; bounds = bnds, init = φ0, frozen = frozen ? 1.0 : nothing))
  x = DesignFields(z; φ = Identity())
  θ = PhysicalFields(x; χ = :φ => (design === :levelset ? interface : Identity()))
  dΩ = design_measures(θ; degree = design === :levelset ? 4 : 2)
  ν, Emin = 0.3, 1e-3
  λ, μ = ν / (1 - ν^2), 1 / (2(1 + ν))
  σ(e) = λ * tr(e) * one(e) + 2μ * e
  E, dE = design === :levelset ? (χ -> Emin + χ * (1 - Emin), χ -> 1 - Emin) : (ρ -> Emin + ρ^3 * (1 - Emin), ρ -> 3ρ^2 * (1 - Emin))
  t = VectorValue(0.0, -1.0)
  a(u, v) = ∫((E ∘ θ.χ) * (ε(v) ⊙ (σ ∘ ε(u))))dΩ.domain
  l(v) = ∫(t ⋅ v)dΓ
  model_u = StaticLinearModel(l, a, U, V, bc; ls = LUSolver(), backend = :compiled)
  adj = AdjointProblem(sensitivity = (u, q) -> (δ -> ∫((dE ∘ θ.χ) * (ε(q) ⊙ (σ ∘ ε(u))) * δ)dΩ.domain))
  S = solve_flag === nothing ? StateOperator(model_u, θ; adjoint = adj) :
      StateOperator(model_u; depends = θ, adjoint = adj, solve = (m, ctx) -> (solve!(m; Assembly = true), solve_flag(ctx)))
  C = Functional((u, θ) -> ∫(t ⋅ u)dΓ, S; dJdu = SelfAdjoint(), scale = :initial)
  Vol = Volume(θ, dΩ; max_fraction = fraction)
  cons = (volume = Vol,)
  if left_cap !== nothing         # the material in the left half, at most left_cap |D|: a second constraint
    left = CellField(x -> x[1] < 1.0 ? 1.0 : 0.0, Ω)
    cap = left_cap * 2.0
    L = Functional(θ -> sum(∫(left * θ.χ)dΩ.domain) - cap, θ; dJdθ = θ -> (χ = δ -> ∫(left * δ)dΩ.domain,), scale = cap)
    cons = (volume = Vol, left = L)
  end
  (; model, z, x, θ, S, C, Vol, dΩ, Vφ, Emin, σ, E, dE, t, dΓ, problem = Optimization(z; objective = C, constraints = cons))
end

# the same cantilever from factories (RebuildCheck)
function rd_factories(; nx = 30, ny = 15)
  model = CartesianDiscreteModel((0, 2, 0, 1), (nx, ny))
  labels = get_face_labeling(model)
  add_tag_from_vertex_filter!(labels, "clamp", model, x -> x[1] < 1e-9)
  add_tag_from_vertex_filter!(labels, "load", model, x -> x[1] > 2 - 1e-9 && abs(x[2] - 0.5) <= 0.1 + 1e-9)
  tag_adjacent_cells!(labels, "load_zone", model, "load")
  design_setup(model) = begin
    Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1, dirichlet_tags = "load_zone")
    z = OptVariables(φ = FieldVariable(Vφ; bounds = (-1.0, 1.0), init = 1.0, frozen = 1.0))
    (; z, x = DesignFields(z; φ = Identity()))
  end
  physics_setup(model, x) = begin
    θ = PhysicalFields(x; χ = :φ => Ersatz())
    dΩ = design_measures(θ; degree = 4)
    Ω = Triangulation(model)
    dΓ = Measure(BoundaryTriangulation(model; tags = "load"), 2)
    bc = BoundaryConditions(BoundaryCondition("clamp", VectorValue(0.0, 0.0)))
    V = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1), bc; conformity = :H1)
    U = TrialFESpace(V, bc, 1.0)
    ν, Emin = 0.3, 1e-3
    λ, μ = ν / (1 - ν^2), 1 / (2(1 + ν))
    σ(e) = λ * tr(e) * one(e) + 2μ * e
    E(χ) = Emin + χ * (1 - Emin)
    t = VectorValue(0.0, -1.0)
    a(u, v) = ∫((E ∘ θ.χ) * (ε(v) ⊙ (σ ∘ ε(u))))dΩ.domain
    S = StateOperator(StaticLinearModel(v -> ∫(t ⋅ v)dΓ, a, U, V, bc; ls = LUSolver(), backend = :compiled), θ;
          adjoint = AdjointProblem(sensitivity = (u, q) -> (δ -> ∫((1 - Emin) * (ε(q) ⊙ (σ ∘ ε(u))) * δ)dΩ.domain)))
    C = Functional((u, θ) -> ∫(t ⋅ u)dΓ, S; dJdu = SelfAdjoint(), scale = :initial)
    (; objective = C, constraints = (volume = Volume(θ, dΩ; max_fraction = 0.5),))
  end
  Optimization(model; design_setup, physics_setup)
end

rd(; kw...) = ReactionDiffusion(; τ = 2e-4, sensitivity = :physical, kw...)

# the oracle and the state of the step at the starting design, as the loop builds them
function rd_start(c, m)
  fo = RDM.build_oracle(m, c.problem)
  for i in 1:OPT.num_constraints(fo)
    RDM._is_linear(fo, i) && RDM._setup_constraint!(fo, i, c.problem.ctx)
  end
  z = copy(get_values(c.z))
  fo, z, OPT.init(m.step, z, fo)
end

# connected components of the cells with χ < 1/2 (Cartesian nx × ny, P0 values in the order of the cells)
function void_components(χ, nx, ny)
  void = reshape(χ .< 0.5, nx, ny)
  seen = falses(nx, ny)
  n = 0
  for j in 1:ny, i in 1:nx
    (void[i, j] && !seen[i, j]) || continue
    n += 1
    stack = [(i, j)]
    seen[i, j] = true
    while !isempty(stack)
      (a, b) = pop!(stack)
      for (p, q) in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1))
        (1 <= p <= nx && 1 <= q <= ny && void[p, q] && !seen[p, q]) || continue
        seen[p, q] = true
        push!(stack, (p, q))
      end
    end
  end
  n
end

cell_means(f, model) = (dΩ = Measure(Triangulation(model), 4); V0 = FESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2);
                        collect(assemble_vector(v -> ∫(f * v)dΩ, V0)) ./ collect(assemble_vector(v -> ∫(v)dΩ, V0)))

# a monitor that checks that the material of the frozen cells stays 1 at every accepted iteration
struct FrozenCheck{T,Z,M}
  θ::T
  zone::Z
  dΩ::M
  ok::Base.RefValue{Bool}
end
function HyperShape.Methods.monitor_value(f::FrozenCheck)
  e = sum(∫(f.zone.χ * (1 - f.θ.χ))f.dΩ)
  abs(e) <= 1e-12 || (f.ok[] = false)
  e
end

@testset "ReactionDiffusion" begin

  @testset "1. φ̌ without frozen values is Δt·1, with no solve" for simplex in (false, true), mass in (:lumped, :consistent)
    c = rd_cantilever(; nx = 20, ny = 10, frozen = false, simplex)
    m = rd(; mass)
    fo, z, s = rd_start(c, m)
    @test all(==(0.1), s.φc) && s.nsolves == 0 && s.nfactor == 1
    @test maximum(abs, s.A * s.φc .- s.m) <= 1e-13 * maximum(s.m)
  end

  @testset "2. φ̌ with frozen values: A⁻¹m; lumped mass: positive, M-matrix, G(λ) non-increasing" for simplex in (false, true)
    c = rd_cantilever(; nx = 20, ny = 10, simplex)
    for mass in (:lumped, :consistent)
      fo, z, s = rd_start(c, rd(; mass))
      @test s.φc ≈ Matrix(s.A) \ s.m rtol = 1e-12
      @test s.nsolves == 1 && s.nfactor == 1
      x, y = [sin(1.3i) for i in eachindex(s.m)], [cos(0.7i) for i in eachindex(s.m)]
      @test dot(s.A * x, y) ≈ dot(x, s.A' * y) rtol = 1e-13
      @test s.A ≈ s.A'                                       # a symmetric operator
      mass === :lumped || continue
      Ad = Matrix(s.A)
      @test minimum(s.φc) > 0
      @test maximum(Ad - Diagonal(diag(Ad))) <= 0
      φs = [sin(0.37i) for i in eachindex(s.m)]
      G = [dot(s.m, clamp.(φs .- λ .* s.φc, -1.0, 1.0)) for λ in range(-25, 25; length = 200)]
      @test all(<=(1e-12), diff(G))
    end
  end

  @testset "3. one step against the algebra by hand (frozen zone, density, exact multiplier)" begin
    c = rd_cantilever(; nx = 20, ny = 10, design = :density)
    m = rd()
    fo, z, s = rd_start(c, m)
    z0 = copy(z)
    U = c.z.blocks[1].space
    dv0 = copy(get_dirichlet_dof_values(U))
    info = OPT.step!(m.step, s, z, fo, c.problem.ctx)
    # by hand
    OPT.objective(fo, z0)
    g = OPT.gradient!(similar(z0), fo, z0)
    Vt = U.space
    dΩ = Measure(Triangulation(c.model), 2)
    K = assemble_matrix((u, v) -> ∫(∇(u) ⋅ ∇(v))dΩ, TrialFESpace(Vt), Vt)
    mv = assemble_vector(v -> ∫(v)dΩ, Vt)
    kd = assemble_vector(v -> ∫(∇(FEFunction(U, zeros(num_free_dofs(U)))) ⋅ ∇(v))dΩ, Vt)
    area = sum(∫(1.0)dΩ)
    τℓ² = 2e-4 * area
    A = Matrix(Diagonal(mv) / 0.1 + τℓ² * K)
    C = area / sum(abs, g)
    φs = z0 .+ A \ (-τℓ² .* (K * z0 .+ kd) .- C .* g)
    φc = A \ mv
    λ = info.multipliers[1]
    @test z ≈ clamp.(φs .- λ .* φc, 0.0, 1.0) atol = 1e-12
    @test info.accepted && info.step == 0.1 && info.change ≈ maximum(abs, z .- z0)
    @test get_dirichlet_dof_values(U) == dv0
    @test abs(OPT.constraint(fo, 1, z)) <= 1e-4
  end

  @testset "4. exact volume with a density: met at every iteration from an infeasible start" begin
    c = rd_cantilever(; nx = 30, ny = 15, design = :density, init = 0.9)
    r = solve!(c.problem, rd(); maxiter = 15, stop = nothing, verbose = 0)
    @test all(v -> abs(v) <= 1e-4, r.history.constraints.volume)
    @test r.start_scaling === nothing
    @test r.state.nvol <= 6 * r.iterations
  end

  @testset "5. the target of the volume goes down max_step per iteration ($(iface))" for iface in (Ersatz(), Step())
    c = rd_cantilever(; nx = 30, ny = 15, interface = iface)
    tol = 1e-4
    dg = 0.05 * 2.0 / get_scale(c.Vol)                     # max_step |D| / s
    r = solve!(c.problem, rd(; volume = ExactVolume(; max_step = 0.05)); maxiter = 14, stop = nothing, verbose = 0)
    g = [1.0; r.history.constraints.volume]                  # solid start: g₀ = (|D| − Vmax)/Vmax = 1
    # with Step() the volume moves by quadrature points and the root ends on the feasible side when no node of the
    # clipped design changes side inside the bracket: below the target by a few cells at most (here 5)
    qtol = iface isa Step ? 5 * (2.0 / (30 * 15)) / get_scale(c.Vol) : tol
    for k in 2:length(g)
      @test g[k-1] - g[k] <= dg + qtol
      @test -qtol <= g[k] - max(0.0, g[k-1] - dg) <= tol
    end
    @test findfirst(v -> v <= tol, g[2:end]) == ceil(Int, 0.5 / 0.05)
    c2 = rd_cantilever(; nx = 30, ny = 15, interface = iface)
    r2 = solve!(c2.problem, rd(; volume = ExactVolume(; max_step = nothing)); maxiter = 2, stop = nothing, verbose = 0)
    @test r2.history.constraints.volume[1] <= tol
  end

  @testset "6. the root of the multiplier" begin
    R = RDM._root
    for dG in (x -> -1.0, nothing)
      x, gx, st = R(x -> 1 - x, dG, 0.0, -10.0, 10.0, 0.1, 1e-12, 60)
      @test st === :ok && abs(x - 1) <= 1e-10
    end
    for dG in (x -> -exp(-x), nothing)                       # smooth; a start outside the bracket
      x, gx, st = R(x -> exp(-x) - 0.5, dG, 100.0, -10.0, 10.0, 0.1, 1e-12, 60)
      @test st === :ok && abs(x - log(2)) <= 1e-10
    end
    x, gx, st = R(x -> clamp(2 - x, -1.0, 1.0), nothing, -5.0, -10.0, 10.0, 0.1, 1e-12, 60)      # piecewise linear
    @test st === :ok && abs(x - 2) <= 1e-10
    x, gx, st = R(x -> x < 0.37 ? 0.5 : -0.5, nothing, 0.0, -10.0, 10.0, 0.1, 1e-6, 60)          # unreachable tolerance
    @test st === :ok && gx == -0.5 && 0.37 <= x <= 0.37 + 1e-8
    @test R(x -> 1.0, nothing, 0.0, -10.0, 10.0, 0.1, 1e-6, 60)[3] === :infeasible               # no sign change
    x, gx, st = R(x -> -1.0, nothing, 3.0, 0.0, 10.0, 0.1, 1e-6, 60)                               # inactive: λ = 0
    @test st === :ok && x == 0.0
    # a volume that the bounds cannot meet (the frozen zone is larger): an error in the first step, with the hint
    c = rd_cantilever(; nx = 20, ny = 10, design = :density, fraction = 0.005)
    @test_throws r"cannot be met.*frozen zones" solve!(c.problem, rd(); maxiter = 2, verbose = 0)
  end

  @testset "6b. the derivative of G(λ): Taylor (order 2 with Ersatz, exact for a linear volume)" begin
    φ0 = x -> 0.3 - hypot(x[1] - 1.0, x[2] - 0.5)
    c = rd_cantilever(; nx = 20, ny = 10, frozen = false, init = x -> clamp(φ0(x), -1.0, 1.0))
    fo, z, s = rd_start(c, rd())
    s.φs .= z                                               # φ* = z; shifts of λ φ̌ = 0.1 λ, far from the bounds
    lb, ub = OPT.bounds(fo)
    G, D = RDM._OracleG(fo, s, lb, ub, 0.0), RDM._OracleDG(fo, s, lb, ub)
    λ0 = 0.5
    g0, d0 = G(λ0), D(λ0)
    @test d0 < 0
    errs = [abs(G(λ0 + h) - g0 - h * d0) for h in (0.04, 0.02, 0.01, 0.005)]
    @test all(>=(1.8), log2.(errs[1:end-1] ./ errs[2:end]))
    cd = rd_cantilever(; nx = 20, ny = 10, design = :density)
    fo, z, s = rd_start(cd, rd())
    s.φs .= z
    a, V = OPT.linear_constraint(fo, 1)
    lb, ub = OPT.bounds(fo)
    G, D = RDM._LinearG(a, V, s, lb, ub, 0.0), RDM._LinearDG(a, s, lb, ub)
    g0, d0 = G(0.5), D(0.5)
    @test all(h -> abs(G(0.5 + h) - g0 - h * d0) <= 1e-12, (0.1, 0.01))
  end

  @testset "7. normalisations" begin
    Cs = map((20, 80)) do n
      model = CartesianDiscreteModel((0, 2, 0, 1), (n, n ÷ 2))
      V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
      dΩ = Measure(Triangulation(model), 4)
      g = collect(assemble_vector(v -> ∫((x -> -(1 + x[1]^2)) * v)dΩ, V))
      m = collect(assemble_vector(v -> ∫(v)dΩ, V))
      @test L1Mean(3.0)(g, m, 2.0) ≈ 3 * L1Mean()(g, m, 2.0)
      @test L2Norm()(g, m, 2.0) ≈ 2.0 / sqrt(sum(g .^ 2 ./ m))
      @test MaxAbs()(g, m, 2.0) ≈ 1 / maximum(abs.(g) ./ m)
      @test (@allocated L1Mean()(g, m, 2.0)) == 0 && (@allocated L2Norm()(g, m, 2.0)) == 0 && (@allocated MaxAbs()(g, m, 2.0)) == 0
      L1Mean()(g, m, 2.0)
    end
    @test Cs[1] ≈ Cs[2] rtol = 1e-3
    @test_throws ArgumentError L1Mean(0.0)
    calls = Ref(0)
    c = rd_cantilever(; nx = 20, ny = 10, design = :density)
    r = solve!(c.problem, rd(; normalize = (g, m, a) -> (calls[] += 1; a / sum(abs, g))); maxiter = 3, stop = nothing, verbose = 0)
    c2 = rd_cantilever(; nx = 20, ny = 10, design = :density)
    r2 = solve!(c2.problem, rd(); maxiter = 3, stop = nothing, verbose = 0)
    @test calls[] == 3 && get_values(r.z) == get_values(r2.z)
  end

  @testset "8. sensitivity = :physical: the derivative weighted by the material ($(iface))" for iface in (Ersatz(), Step())
    φ0 = x -> 0.3 - hypot(x[1] - 1.0, x[2] - 0.5) + 0.2       # a hole: an interface in the design
    c = rd_cantilever(; nx = 20, ny = 10, frozen = false, interface = iface, init = x -> clamp(φ0(x), -1.0, 1.0))
    w0 = 0.01
    m = rd(; void_weight = w0)
    fo, z, s = rd_start(c, m)
    OPT.objective(fo, z)
    RDM._rd_gradient!(s.g, m.step, s, fo, z)
    @test c.problem.ctx.sensitivity === :design            # back to :design after the step's sweep
    u = get_state(c.S)
    χ = c.θ.χ
    ghand = -collect(assemble_vector(v -> ∫((w0 + (1 - w0) * χ) * (c.dE ∘ χ) * (ε(u) ⊙ (c.σ ∘ ε(u))) * v)c.dΩ.domain, c.Vφ)) ./
            get_scale(c.C)
    @test s.g ≈ ghand rtol = 1e-10
    if iface isa Ersatz
      md = ReactionDiffusion(; τ = 2e-4, sensitivity = :design)
      # (an oracle of its own: the snapshot of fo holds the gradient of the step, the :physical one)
      g1 = RDM._rd_gradient!(similar(z), md.step, s, RDM.build_oracle(md, c.problem), z)
      # the exact derivative through Ersatz() lives on the nodes of the cut cells; the :physical one everywhere
      cut = Set{Int}()
      for ids in get_cell_dof_ids(c.Vφ)
        v = z[ids]
        (any(<(0), v) && any(>=(0), v)) && union!(cut, ids)
      end
      @test all(i -> i in cut || g1[i] == 0, eachindex(g1)) && any(!iszero, g1)
      @test count(!iszero, s.g) > length(cut)
      # the exact derivative, after a :physical sweep (steps below the default: the larger ones cross the kinks of Ersatz)
      @test check_consistency((objective = c.C,); h0 = 1e-3).passed
    end
  end

  @testset "8b. without an interface map both sensitivities are the same bit for bit" begin
    gs = map((:physical, :design)) do sens
      c = rd_cantilever(; nx = 20, ny = 10, design = :density)
      m = ReactionDiffusion(; τ = 2e-4, sensitivity = sens)
      r = solve!(c.problem, m; maxiter = 5, stop = nothing, verbose = 0)
      copy(get_values(r.z))
    end
    @test gs[1] == gs[2]
  end

  @testset "9. the cantilever 60×30: holes, bounds, frozen zone, volume, better than uniform" begin
    nx, ny = 60, 30
    # a density by OC on the same mesh (P1, SIMP p = 3, no filter), for reference
    cd = rd_cantilever(; nx, ny, design = :density)
    rd_oc = solve!(cd.problem, DensityMethod(OC()); maxiter = 150, verbose = 0)
    # uniform material χ = 0.5: compliance 1/E(0.5) times the solid one (scaled by the solid one)
    Junif = 1 / (1e-3 + 0.5 * (1 - 1e-3))
    for iface in (Ersatz(), Step())
      c = rd_cantilever(; nx, ny, interface = iface)
      frozen_ok = Ref(true)
      zone = c.z.blocks[1].zone
      mon = (frozen = FrozenCheck(c.θ, zone, c.dΩ.domain, frozen_ok),)
      r = solve!(c.problem, rd(); maxiter = 220, stop = nothing, verbose = 0, monitors = mon)
      J = r.objective
      @test J < Junif
      # in absolute value (the scales are the first compliances, solid and grey); measured: Ersatz 2.371, Step 2.390,
      # OC 2.685 (the Mac of the lab, 4 threads, Julia 1.13.0, 2026-11-23)
      @test J * get_scale(c.C) <= 1.1 * rd_oc.objective * get_scale(cd.C)
      @test void_components(cell_means(c.θ.χ, c.model), nx, ny) >= 1
      @test all(v -> -1 <= v <= 1, get_values(r.z)) && frozen_ok[]
      @test abs(r.history.constraints.volume[end]) <= (iface isa Step ? 1e-3 : 1e-4)
    end
    c = rd_cantilever(; nx, ny, design = :density)
    grey(z) = (ρ = cell_means(c.θ.χ, c.model); 4 * sum(ρ .* (1 .- ρ)) / length(ρ))
    g0 = grey(c.z)
    r = solve!(c.problem, rd(); maxiter = 100, verbose = 0)
    @test grey(c.z) < g0
  end

  @testset "10. stop on the change with the volume met, never during the descent" begin
    # the mechanism of the default stop with a larger tolerance (measured: converged at iteration 91 of 300, the
    # volume reached at 50); the iteration at which ‖Δφ‖∞ < 1e-2 is sensitive to perturbations of the volume
    c = rd_cantilever(; nx = 30, ny = 15)
    r = solve!(c.problem, rd(); maxiter = 300, verbose = 0, stop = StopCriteria(change_tol = 3e-2))
    @test r.status === :converged && r.reason === :change_tol
    @test abs(r.history.constraints.volume[end]) <= 1e-4 && r.history.change[end] < 3e-2
    k0 = findfirst(v -> v <= 1e-4, r.history.constraints.volume)
    @test r.iterations > k0
    @test all(k -> r.history.change[k] >= 3e-2 || r.history.constraints.volume[k] > 1e-4, 2:r.iterations-1)
    @test RDM.default_stop(rd().step).change_tol == 1e-2
  end

  @testset "11. rejected steps: Δt halved, then back; exhausted attempts" begin
    c = rd_cantilever(; nx = 20, ny = 10, solve_flag = ctx -> !(ctx.iteration == 3 && ctx.attempt == 1))
    r = solve!(c.problem, rd(); maxiter = 5, stop = nothing, verbose = 0)
    h = r.history
    @test h.attempts[3] == 2 && h.step[3] == 0.05 && h.step[4] == 0.1 && h.step[2] == 0.1
    @test r.state.nfactor == 3                            # init, the reduction and the return to Δt
    # the target of the volume goes down in proportion to Δt in the shrunk step (0.02 → 0.01), and back after it
    g = [1.0; h.constraints.volume]
    @test abs((g[3] - g[4]) - 0.01) <= 1e-4 && abs((g[4] - g[5]) - 0.02) <= 1e-4 && abs((g[2] - g[3]) - 0.02) <= 1e-4
    dir = mktempdir()
    c = rd_cantilever(; nx = 20, ny = 10, solve_flag = ctx -> ctx.iteration < 3)
    @test_throws OptimizationFailure solve!(c.problem, rd(); maxiter = 5, stop = nothing, verbose = 0,
                                            checkpoint = Checkpoint(filepath = dir, every = 10))
    @test isfile(joinpath(dir, "checkpoint.toml")) && load_checkpoint(dir).iteration == 2
  end

  @testset "12. checkpoint and resume, and a rebuild, bit for bit (in the descent of the volume)" begin
    ref = rd_cantilever(; nx = 30, ny = 15)
    r20 = solve!(ref.problem, rd(); maxiter = 20, stop = nothing, verbose = 0)
    dir = mktempdir()
    a = rd_cantilever(; nx = 30, ny = 15)
    solve!(a.problem, rd(); maxiter = 10, stop = nothing, verbose = 0, checkpoint = Checkpoint(filepath = dir, every = 10))
    b = rd_cantilever(; nx = 30, ny = 15)
    rb = solve!(b.problem, rd(); maxiter = 20, stop = nothing, verbose = 0, resume = dir)
    @test get_values(rb.z) == get_values(r20.z) && rb.history.objective == r20.history.objective
    @test r20.history.constraints.volume[20] > 0.1          # still going down
    p1, p2 = rd_factories(), rd_factories()
    m = rd()
    f1 = solve!(p1, m; maxiter = 15, stop = nothing, verbose = 0)
    f2 = solve!(p2, m; maxiter = 15, stop = nothing, verbose = 0, discretization = RDM.RebuildCheck(every = 6))
    @test get_values(f1.z) == get_values(f2.z) && f1.history.objective == f2.history.objective
  end

  @testset "13. a second constraint in the augmented Lagrangian, with the native volume" begin
    c = rd_cantilever(; nx = 30, ny = 15, left_cap = 0.2)
    r = solve!(c.problem, rd(); maxiter = 250, stop = nothing, verbose = 0)
    @test r.history.violations.volume[end] <= 1e-4
    @test r.history.violations.left[end] <= 1e-3
    @test isnan(r.history.penalties.volume[end]) && r.history.penalties.left[end] == 1.0
  end

  @testset "14. preset :kim2020" begin
    m = ReactionDiffusion(:kim2020; τ = 5e-4, sensitivity = :physical)
    @test m.step.Δt == 0.1 && m.step.normalize isa L2Norm && m.step.mass === :consistent && m.step.volume === nothing
    u = m.lagrangian.update
    @test m.lagrangian.native === () && (u.μ0, u.growth, u.window, u.progress, u.μmax) == (0.2, 1.025, 1, Inf, 2.0)
    @test RDM.method_label(m) == "ReactionDiffusion(:kim2020)"
    m2 = ReactionDiffusion(:kim2020; τ = 5e-4, sensitivity = :physical, mass = :lumped)
    @test m2.step.mass === :lumped && m2.sources[:mass] === :given && m2.sources[:normalize] === :preset
    c = rd_cantilever(; nx = 20, ny = 10, design = :density)
    io = IOBuffer()
    r = solve!(c.problem, m2; maxiter = 3, stop = nothing, verbose = 1, io)
    s = String(take!(io))
    @test occursin("every constraint goes to the augmented Lagrangian", s) && occursin("augmented Lagrangian: Bertsekas", s)
    @test occursin("lumped mass (given, replaces :kim2020)", s) && occursin("L2Norm() (:kim2020)", s)
    @test r.state.kind === :none && isfinite(r.history.multipliers.volume[1])   # the λ of the augmented Lagrangian
  end

  @testset "15. errors" begin
    @test_throws r"τ is mandatory" ReactionDiffusion(; sensitivity = :physical)
    @test_throws r"sensitivity is mandatory" ReactionDiffusion(; τ = 1e-4)
    @test_throws r"sensitivity must be" ReactionDiffusion(; τ = 1e-4, sensitivity = :exact)
    @test_throws r"unknown preset :otomori2015; the presets are :kim2020" ReactionDiffusion(:otomori2015; τ = 1e-4, sensitivity = :physical)
    @test_throws r"native = \(\)" ReactionDiffusion(; τ = 1e-4, sensitivity = :physical, volume = nothing)
    @test_throws r"unknown keyword phi_bc" ReactionDiffusion(; τ = 1e-4, sensitivity = :physical, phi_bc = :neumann)
    @test_throws r"mass must be" rd(; mass = :diagonal)
    @test_throws r"Δt must be" rd(; Δt = 0.0)
    # P0, infinite bounds, Step() and Ersatz() with :design
    c = rd_cantilever(; nx = 10, ny = 5, design = :density, order = 0)
    @test_throws r"nodal P1" solve!(c.problem, rd(); maxiter = 1, verbose = 0)
    c = rd_cantilever(; nx = 10, ny = 5, design = :density, bounds = (0.0, Inf))
    @test_throws r"bounds of z must be finite" solve!(c.problem, rd(); maxiter = 1, verbose = 0)
    c = rd_cantilever(; nx = 10, ny = 5, interface = Step())
    @test_throws r"Step\(\), which has no derivative" solve!(c.problem, ReactionDiffusion(; τ = 1e-4, sensitivity = :design); maxiter = 1, verbose = 0)
    c = rd_cantilever(; nx = 10, ny = 5)
    @test_throws r"starting design has none" solve!(c.problem, ReactionDiffusion(; τ = 1e-4, sensitivity = :design); maxiter = 1, verbose = 0)
    # two blocks
    model = CartesianDiscreteModel((0, 1, 0, 1), (4, 4))
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    z = OptVariables(a = FieldVariable(V; bounds = (0.0, 1.0), init = 0.5), b = FieldVariable(V; bounds = (0.0, 1.0), init = 0.5))
    θ = PhysicalFields(DesignFields(z; a = Identity(), b = Identity()); a = Identity(), b = Identity())
    dΩ = Measure(Triangulation(model), 2)
    F = Functional(θ -> sum(∫(θ.a + θ.b)dΩ), θ; dJdθ = θ -> (a = δ -> ∫(δ)dΩ, b = δ -> ∫(δ)dΩ), scale = 1.0)
    @test_throws r"one design field" solve!(Optimization(z; objective = F), rd(); maxiter = 1, verbose = 0)
    # DensityMethod still refuses the interface maps
    c = rd_cantilever(; nx = 10, ny = 5)
    @test_throws r"interface map" solve!(c.problem, DensityMethod(OC()); maxiter = 1, verbose = 0)
  end

  @testset "16. allocations and counts per iteration" begin
    c = rd_cantilever(; nx = 20, ny = 10, design = :density)
    m = rd()
    fo, z, s = rd_start(c, m)
    RDM._rd_rhs!(s, z, 0.7)
    @test (@allocated RDM._rd_rhs!(s, z, 0.7)) == 0
    a, V = OPT.linear_constraint(fo, 1)
    lb, ub = OPT.bounds(fo)
    G, D = RDM._LinearG(a, V, s, lb, ub, 0.0), RDM._LinearDG(a, s, lb, ub)
    G(0.1); D(0.1)
    @test (@allocated G(0.1)) == 0 && (@allocated D(0.1)) == 0
    RDM._rd_solve!(s.δ, m.step, s, s.r)
    @test (@allocated RDM._rd_solve!(s.δ, m.step, s, s.r)) <= 512      # the substitution of UMFPACK
    c = rd_cantilever(; nx = 20, ny = 10, design = :density)
    r = solve!(c.problem, m; maxiter = 10, stop = nothing, verbose = 0)
    @test all(==(1), r.history.forward[2:end]) && all(==(0), r.history.adjoint)
    @test r.state.nsolves == 1 + 10 && r.state.nfactor == 1           # φ̌ and one solve per iteration
    # a solve function of the user (here the same LU): the same designs
    calls = Ref(0)
    lusolve = (A, b, x, ctx, stale) -> (calls[] += 1; x .= A \ b)
    c2 = rd_cantilever(; nx = 20, ny = 10, design = :density)
    r2 = solve!(c2.problem, rd(; solve = lusolve); maxiter = 10, stop = nothing, verbose = 0)
    @test calls[] == 11
    @test get_values(r2.z) ≈ get_values(r.z) rtol = 1e-10
  end

  @testset "17. failures and limits: a violated linear target, values not finite, the root, OptContext" begin
    # a linear volume above its target (max_step) is not met: no convergence on the change (it was :converged at 12)
    c = rd_cantilever(; nx = 40, ny = 20, design = :density, init = 0.9)
    r = solve!(c.problem, rd(; Δt = 0.01, volume = ExactVolume(; max_step = 0.0005)); maxiter = 40, verbose = 0)
    @test r.status !== :converged || r.history.constraints.volume[end] <= 1e-4
    @test r.status === :maxiter && r.history.constraints.volume[end] > 0.5
    # a normalisation that returns NaN, a solve of the user that returns NaN: failures with a checkpoint
    for kw in ((normalize = (g, m, a) -> NaN,), (solve = (A, b, x, ctx, stale) -> (x .= NaN),))
      dir = mktempdir()
      c = rd_cantilever(; nx = 20, ny = 10, frozen = false)
      @test_throws OptimizationFailure solve!(c.problem, rd(; kw...); maxiter = 3, stop = nothing, verbose = 0,
                                              checkpoint = Checkpoint(filepath = dir, every = 10))
      @test isfile(joinpath(dir, "checkpoint.toml"))
    end
    # a zero sensitivity: diffusion alone (C = 0)
    c = rd_cantilever(; nx = 20, ny = 10, design = :density)
    m = rd()
    fo, z, s = rd_start(c, m)
    fill!(s.g, 0.0)
    @test RDM._normalization(m.step, s) == 0.0
    # the evaluations run out before a feasible trial: the feasible bound of the bracket, not G > 0
    x, gx, st = RDM._root(x -> 1 - x / 100, nothing, 0.0, -10.0, 1000.0, 0.1, 1e-6, 4)
    @test st === :ok && gx <= 0 && x == 1000.0
    @test RDM._root(x -> NaN, nothing, 0.0, -10.0, 10.0, 0.1, 1e-6, 4)[3] === :nonfinite
    # the positional constructor of OptContext with five fields
    ctx = OptContext(3, 1, 2, :optimization, (β = 2.0,))
    @test ctx.iteration == 3 && ctx.sensitivity === :design && ctx.void_weight == 0.0
    @test_throws ArgumentError ExactVolume(; max_step = 2.0)
    @test_throws ArgumentError ExactVolume(; tol = 0.0)
    @test_throws ArgumentError ExactVolume(; maxiter = 1)
    # the whole step with a linear volume allocates no more than the substitution of the step
    stepalloc(st, s, z, fo, ctx, z0) = (copyto!(z, z0); @allocated OPT.step!(st, s, z, fo, ctx))
    z0 = copy(z)
    OPT.step!(m.step, s, z, fo, c.problem.ctx)
    stepalloc(m.step, s, z, fo, c.problem.ctx, z0)
    a = stepalloc(m.step, s, z, fo, c.problem.ctx, z0)
    RDM._rd_solve!(s.δ, m.step, s, s.r)
    @test a <= @allocated RDM._rd_solve!(s.δ, m.step, s, s.r)
  end

  @testset "18. options: diffusion, ls, Parameter, :design, consistent mass, 3D, normalisations" begin
    # a Robin term in the diffusion and no frozen values: φ̌ by a solve
    c = rd_cantilever(; nx = 20, ny = 10, design = :density, frozen = false)
    dΩ2 = Measure(Triangulation(c.model), 2)
    dΓc = Measure(BoundaryTriangulation(c.model; tags = "clamp"), 2)
    m = rd(; diffusion = (φ, v) -> ∫(∇(φ) ⋅ ∇(v))dΩ2 + ∫(10.0 * φ * v)dΓc)
    fo, z, s = rd_start(c, m)
    @test !s.unit && s.nsolves == 1 && minimum(s.φc) < 0.1
    @test s.φc ≈ Matrix(s.A) \ s.m rtol = 1e-12
    r = solve!(c.problem, m; maxiter = 4, stop = nothing, verbose = 0)
    @test all(v -> abs(v) <= 1e-4, r.history.constraints.volume)
    # another linear solver: the same designs
    rs = map((LUSolver(), BackslashSolver())) do ls
      c = rd_cantilever(; nx = 20, ny = 10, design = :density)
      copy(get_values(solve!(c.problem, rd(; ls); maxiter = 5, stop = nothing, verbose = 0).z))
    end
    @test rs[1] ≈ rs[2] rtol = 1e-10
    # τ and Δt continued (Parameter): one factorisation more at the jump
    pτ, pΔt = Parameter(2e-4), Parameter(0.1)
    c = rd_cantilever(; nx = 20, ny = 10)
    r = solve!(c.problem, ReactionDiffusion(; τ = pτ, Δt = pΔt, sensitivity = :physical); maxiter = 6, stop = nothing, verbose = 0,
               continuation = Continuation(τ = pτ => [2e-4, 1e-3], Δt = pΔt => [0.1, 0.05]; every = 3))
    @test r.state.nfactor == 2 && r.state.τ == 1e-3 && r.history.step == [0.1, 0.1, 0.1, 0.05, 0.05, 0.05]
    # sensitivity = :design with Ersatz() from a design with a hole
    φ0 = x -> clamp(0.3 - hypot(x[1] - 1.0, x[2] - 0.5) + 0.2, -1.0, 1.0)
    c = rd_cantilever(; nx = 20, ny = 10, init = φ0)
    r = solve!(c.problem, ReactionDiffusion(; τ = 2e-4, sensitivity = :design); maxiter = 8, stop = nothing, verbose = 0)
    g = r.history.constraints.volume
    @test all(isfinite, r.history.objective) && all(k -> abs(g[k] - max(0.0, g[k-1] - 0.02)) <= 1e-4, 2:8)
    # consistent mass with a frozen zone
    c = rd_cantilever(; nx = 20, ny = 10)
    r = solve!(c.problem, rd(; mass = :consistent); maxiter = 5, stop = nothing, verbose = 0)
    @test all(k -> abs(r.history.constraints.volume[k] - (1 - 0.02k)) <= 1e-4, 1:5)
    # L2Norm and MaxAbs in a run
    for n in (L2Norm(), MaxAbs())
      c = rd_cantilever(; nx = 20, ny = 10, design = :density)
      r = solve!(c.problem, rd(; normalize = n); maxiter = 3, stop = nothing, verbose = 0)
      @test all(isfinite, r.history.objective) && all(v -> abs(v) <= 1e-4, r.history.constraints.volume)
    end
    # 3D: a level set through Ersatz() on hexahedra, ℓ² = |D|^(2/3)
    model = CartesianDiscreteModel((0, 2, 0, 1, 0, 1), (8, 4, 4))
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    z = OptVariables(φ = FieldVariable(V; bounds = (-1.0, 1.0), init = 1.0))
    θ = PhysicalFields(DesignFields(z; φ = Identity()); χ = :φ => Ersatz())
    dΩ = design_measures(θ; degree = 2).domain
    w = CellField(x -> 1 + x[1], Triangulation(model))
    F = Functional(θ -> sum(∫(w * (1 - θ.χ))dΩ), θ; dJdθ = θ -> (χ = δ -> ∫(-w * δ)dΩ,), scale = 1.0)
    r = solve!(Optimization(z; objective = F, constraints = (volume = Volume(θ, dΩ; max_fraction = 0.5),)), rd();
               maxiter = 5, stop = nothing, verbose = 0)
    @test r.state.ℓ2 ≈ 2.0^(2 / 3) && all(k -> abs(r.history.constraints.volume[k] - (1 - 0.02k)) <= 1e-4, 1:5)
  end

  @testset "banner and events" begin
    c = rd_cantilever(; nx = 20, ny = 10, interface = Step())
    io = IOBuffer()
    solve!(c.problem, rd(); maxiter = 2, verbose = 1, io)
    s = String(take!(io))
    @test occursin("method: sensitivity = :physical (void weight 0.001 (default))", s)
    @test occursin("lumped mass (default)", s) && occursin("LUSolver (default)", s) && occursin("L1Mean(1) (default)", s)
    @test occursin("exact multiplier (Illinois, tol 1e-04), target lowered 1 % of |D| per iteration (default)", s)
    @test occursin("with Step() the change criterion is rarely met", s)
    @test occursin("the starting design violates the constraint volume by 1 (scaled); it is not made feasible", s)
    @test occursin("reaction–diffusion: 3 solves of the step, 1 factorisation,", s)
    @test occursin("stop: change < 0.01 (default)", s)
  end
end
