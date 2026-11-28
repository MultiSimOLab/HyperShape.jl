# ReactionDiffusion with the double-well reaction (DoubleWell): the pointwise reaction, one substep against the algebra
# by hand, the profile of the interface, the exact volume inside the tilt, Δt from Gershgorin, allocations and counts,
# the cantilever and the bridge of Takezawa et al. (2010), presets, errors, the grey measure, the settling,
# checkpoints, triangles and 3D. Uses the helpers of reaction_diffusion.jl (rd_cantilever, rd_start, …).
using Test
using LinearAlgebra
using SparseArrays
using Gridap
using Gridap.FESpaces
using HyperFEM
using HyperShape
using HyperShape.Functionals: get_scale

isdefined(@__MODULE__, :bridge) || include(joinpath(@__DIR__, "..", "..", "validation", "common", "common.jl"))

# a density cantilever (bounds (0, 1), SIMP through E) with holes from hole_seed (an interface of 2 cells in the seed)
function dw_cantilever(; nx = 40, ny = 20, frozen = true, simplex = false, radius = 0.1, spacing = 0.4, init = nothing,
                      symmetry = nothing, kw...)
  if init === nothing
    c0 = rd_cantilever(; nx, ny, design = :density, frozen, simplex)
    init = hole_seed(c0.Vφ; bounds = (0.0, 1.0), radius, spacing, exclude = "clamp", connect = ("clamp", "load"),
                     frozen = frozen ? 1.0 : nothing, symmetry, width = 2 * (2 / nx))      # an interface of 2 cells
  end
  rd_cantilever(; nx, ny, design = :density, frozen, simplex, init, kw...)
end
# the density cantilever of dw_cantilever from factories (a rebuild of the discretisation, RebuildCheck)
function dw_factories(; nx = 40, ny = 20)
  model = CartesianDiscreteModel((0, 2, 0, 1), (nx, ny))
  labels = get_face_labeling(model)
  add_tag_from_vertex_filter!(labels, "clamp", model, x -> x[1] < 1e-9)
  add_tag_from_vertex_filter!(labels, "load", model, x -> x[1] > 2 - 1e-9 && abs(x[2] - 0.5) <= 0.1 + 1e-9)
  tag_adjacent_cells!(labels, "load_zone", model, "load")
  design_setup(model) = begin
    Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1, dirichlet_tags = "load_zone")
    φ0 = hole_seed(Vφ; bounds = (0.0, 1.0), radius = 0.1, spacing = 0.4, exclude = "clamp", connect = ("clamp", "load"),
                   frozen = 1.0, width = 2 * (2 / nx))
    z = OptVariables(φ = FieldVariable(Vφ; bounds = (0.0, 1.0), init = φ0, frozen = 1.0))
    (; z, x = DesignFields(z; φ = Identity()))
  end
  physics_setup(model, x) = begin
    θ = PhysicalFields(x; χ = :φ => Identity())
    dΩ = design_measures(θ; degree = 2)
    Ω = Triangulation(model)
    dΓ = Measure(BoundaryTriangulation(model; tags = "load"), 2)
    bc = BoundaryConditions(BoundaryCondition("clamp", VectorValue(0.0, 0.0)))
    V = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1), bc; conformity = :H1)
    U = TrialFESpace(V, bc, 1.0)
    ν, Emin = 0.3, 1e-3
    λ, μ = ν / (1 - ν^2), 1 / (2(1 + ν))
    σ(e) = λ * tr(e) * one(e) + 2μ * e
    E(ρ) = Emin + ρ^3 * (1 - Emin)
    dE(ρ) = 3ρ^2 * (1 - Emin)
    t = VectorValue(0.0, -1.0)
    a(u, v) = ∫((E ∘ θ.χ) * (ε(v) ⊙ (σ ∘ ε(u))))dΩ.domain
    S = StateOperator(StaticLinearModel(v -> ∫(t ⋅ v)dΓ, a, U, V, bc; ls = LUSolver(), backend = :compiled), θ;
          adjoint = AdjointProblem(sensitivity = (u, q) -> (δ -> ∫((dE ∘ θ.χ) * (ε(q) ⊙ (σ ∘ ε(u))) * δ)dΩ.domain)))
    C = Functional((u, θ) -> ∫(t ⋅ u)dΓ, S; dJdu = SelfAdjoint(), scale = :initial)
    (; objective = C, constraints = (volume = Volume(θ, dΩ; max_fraction = 0.5),))
  end
  Optimization(model; design_setup, physics_setup)
end
# a physical diffusion, the same on every mesh of these tests: κ = τ ℓ² = 4e-4 (ℓ² = |D| = 2), ξ = √(κ/2W) ≈ 0.028
const DW_τ = 2e-4
# references of the CI cases (Mac M5 Max, Julia 1.13): the cantilever 100×50 and the bridge 100×50 with κ = 4e-5
# (measured 2.5371 at 67 iterations; the bridge: 23.3366 at 100 iterations, not settled, 23.4039 with :takezawa2010, 23.2632 with
# the default stop)
const JCA100_LO, JCA100_HI = 2.50, 2.58
const JBR100_LO, JBR100_HI = 23.0, 23.8
const JBR100T_LO, JBR100T_HI = 23.0, 23.8
dw(; kw...) = ReactionDiffusion(; τ = DW_τ, reaction = DoubleWell(; kw...))

# the back-substitutions of a run with the well from iteration k + 1 on (n substeps before the settling, 1 after it)
count_substeps(r, k) = sum(i -> i <= r.state.settle_iter ? r.state.nsub : 1, k+1:r.iterations)

# the semi-implicit update of Takezawa et al. (2010, Eqs. 44–45) solved by hand: φ⁺ = a + Δt φ⁺(1 − φ) r when r ≤ 0,
# φ⁺ = a + Δt φ (1 − φ⁺) r when r > 0 (a the diffused value, r = 4W(φ − ½) − 30 G φ(1 − φ))
function takezawa_by_hand(φ, a, G, Δt, W)
  r = 4W * (φ - 0.5) - 30G * φ * (1 - φ)
  r <= 0 ? a / (1 - Δt * (1 - φ) * r) : (a + Δt * φ * r) / (1 + Δt * φ * r)
end

# a problem without sensitivity and without constraints: the Allen–Cahn evolution of the well alone
function dw_flat(model, φ0; τ)
  V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
  z = OptVariables(φ = FieldVariable(V; bounds = (0.0, 1.0), init = collect(get_free_dof_values(interpolate(φ0, V)))))
  θ = PhysicalFields(DesignFields(z; φ = Identity()); χ = :φ => Identity())
  dΩ = Measure(Triangulation(model), 2)
  F = Functional(θ -> 0.0 * sum(∫(θ.χ)dΩ), θ; dJdθ = θ -> (χ = δ -> ∫(0.0 * δ)dΩ,), scale = 1.0)
  m = ReactionDiffusion(; τ, reaction = DoubleWell(settle = false), volume = nothing,
                        lagrangian = AugmentedLagrangian(; native = ()))
  (; V, z, problem = Optimization(z; objective = F), m)
end

@testset "DoubleWell" begin

  @testset "1. the pointwise reaction: bounded, fixed points, monotone in G, Eqs. 44–45 by hand" begin
    W = 0.25
    for (φ, a, G, Δt) in ((0.3, 0.35, 2.0, 5.0), (0.8, 0.7, -3.0, 0.1), (0.5, 0.5, 0.0, 1.0), (0.1, 0.0, -40.0, 100.0))
      @test RDM._dw_point(φ, a, G, Δt, W, 0.0, 1.0) ≈ clamp(takezawa_by_hand(φ, a, G, Δt, W), 0, 1) rtol = 1e-14
    end
    # on another interval: the same reaction on φ̂ = (φ − lb)/(ub − lb)
    @test RDM._dw_point(-0.4, -0.3, 2.0, 5.0, W, -1.0, 1.0) ≈ -1 + 2 * takezawa_by_hand(0.3, 0.35, 2.0, 5.0, W) rtol = 1e-14
    k = 0
    bad = 0
    for i in 1:10_000
      φ, a = mod(0.6180339887i, 1.0), mod(0.4142135623i + 0.1, 1.0)
      Δt = 10.0^(mod(0.7320508i, 1.0) * 6 - 3)                # 1e-3 … 1e3
      Gs = sort([20 * sin(1.3i + j) for j in 1:6])
      R = [RDM._dw_point(φ, a, G, Δt, W, 0.0, 1.0) for G in Gs]
      bad += !(all(r -> 0 <= r <= 1, R) && all(<=(1e-15), diff(R)))
      k += 1
    end
    @test k == 10_000 && bad == 0
    for G in (-1e3, -1.0, 0.0, 1.0, 1e3), Δt in (1e-3, 1.0, 1e3)
      @test RDM._dw_point(0.0, 0.0, G, Δt, W, 0.0, 1.0) == 0.0 && RDM._dw_point(1.0, 1.0, G, Δt, W, 0.0, 1.0) == 1.0
    end
  end

  @testset "2. one substep against the algebra by hand (frozen zone)" begin
    c = dw_cantilever(; nx = 20, ny = 10, init = x -> hypot(x[1] - 1, x[2] - 0.5) < 0.3 ? 0.1 : 0.9)
    m = dw(; substeps = 1)
    fo, z, s = rd_start(c, m)
    z0 = copy(z)
    U = c.z.blocks[1].space
    dv0 = copy(get_dirichlet_dof_values(U))
    info = OPT.step!(m.step, s, z, fo, c.problem.ctx)
    OPT.objective(fo, z0)
    g = OPT.gradient!(similar(z0), fo, z0)
    Vt = U.space
    dΩ = Measure(Triangulation(c.model), 2)
    K = assemble_matrix((u, v) -> ∫(∇(u) ⋅ ∇(v))dΩ, TrialFESpace(Vt), Vt)
    mv = assemble_vector(v -> ∫(v)dΩ, Vt)
    kd = assemble_vector(v -> ∫(∇(FEFunction(U, zeros(num_free_dofs(U)))) ⋅ ∇(v))dΩ, Vt)
    area = 2.0
    c2 = DW_τ * area
    Δtlim = 2 / maximum(vec(sum(abs.(K), dims = 2)) .* c2 ./ mv)
    @test s.Δtlim ≈ Δtlim rtol = 1e-12
    @test s.Δt ≈ 4 / 3 * Δtlim rtol = 1e-12
    Δt = s.Δt
    ψ = Matrix(Diagonal(mv) / Δt + c2 * K) \ (mv .* z0 ./ Δt .- c2 .* kd)
    C = area / sqrt(sum(g .^ 2 ./ mv))                        # L2Norm()
    gn = 10.0 .* C .* g ./ mv
    λ = info.multipliers[1]
    @test z ≈ [clamp(takezawa_by_hand(z0[i], ψ[i], gn[i] + λ, Δt, 0.25), 0, 1) for i in eachindex(z0)] atol = 1e-12
    @test get_dirichlet_dof_values(U) == dv0
    # the target of the first iteration: 1 % of |D| below the start (scaled units: |D|/Vmax = 2)
    g0 = OPT.constraint(fo, 1, z0)
    @test abs(OPT.constraint(fo, 1, z) - (g0 - 0.01 * 2.0)) <= VT
  end

  @testset "3. the well alone: a flat interface keeps its profile and place; a disc shrinks" begin
    # strip 2 × 0.04, 200 × 4 cells, ξ = 2h: the profile 1/(1 + e^{−x/ξ}) is a steady state
    h, ξ = 0.01, 0.02
    κ = 2 * 0.25 * ξ^2
    model = CartesianDiscreteModel((0, 2, 0, 0.04), (200, 4))
    prof = x -> 1 / (1 + exp(-(x[1] - 1) / ξ))
    p = dw_flat(model, prof; τ = κ / 0.08)
    r = solve!(p.problem, p.m; maxiter = 20, stop = nothing, verbose = 0)
    φ = get_values(r.z)
    nodal(f) = collect(get_free_dof_values(interpolate(f, p.V)))
    xs, ys = nodal(x -> x[1]), nodal(x -> x[2])
    @test maximum(abs, φ .- nodal(prof)) <= 0.03
    # the place of the interface: the x where φ = ½ on the bottom row (linear interpolation)
    row = sort(findall(<(1e-9), ys); by = j -> xs[j])
    i = findfirst(j -> φ[row[j]] >= 0.5, eachindex(row))
    x½ = xs[row[i-1]] + (0.5 - φ[row[i-1]]) / (φ[row[i]] - φ[row[i-1]]) * h
    @test abs(x½ - 1) <= 0.05h
    # a solid disc of radius 0.3 in the unit square shrinks by curvature (Allen–Cahn: the area decreases)
    model = CartesianDiscreteModel((0, 1, 0, 1), (40, 40))
    disc = x -> 1 / (1 + exp((hypot(x[1] - 0.5, x[2] - 0.5) - 0.3) / 0.025))
    p = dw_flat(model, disc; τ = 2 * 0.25 * 0.025^2)
    a0 = sum(get_values(p.z) .>= 0.5)
    r = solve!(p.problem, p.m; maxiter = 30, stop = nothing, verbose = 0)
    @test sum(get_values(r.z) .>= 0.5) < a0
  end

  @testset "4. exact volume at every iteration, the descent of max_step, λ ≥ 0 at the target ($(frozen ? "frozen zone" : "no frozen zone"))" for frozen in (true, false)
    c = dw_cantilever(; nx = 40, ny = 20, frozen)
    r = solve!(c.problem, dw(); maxiter = 40, stop = nothing, verbose = 0)
    g = r.history.constraints.volume
    U = c.z.blocks[1].space
    fo, z, s = rd_start(dw_cantilever(; nx = 40, ny = 20, frozen), dw())
    gs = OPT.constraint(fo, 1, z)                              # the start
    gg = [gs; g]
    dg = 0.01 * 2.0                                            # 1 % of |D| = 2, scale V_max = 1
    @test gs > 0.3
    @test all(k -> gg[k-1] - dg > 0 ? abs(gg[k] - (gg[k-1] - dg)) <= VT : gg[k] <= VT, 2:length(gg))
    @test g[end] <= VT
    λ = r.history.multipliers.volume
    @test all(k -> λ[k] >= 0, findall(<=(VT), g[1:end-1]) .+ 1)
    @test all(v -> 0 <= v <= 1, get_values(r.z))
    frozen && @test all(==(1.0), get_dirichlet_dof_values(U))   # the frozen values untouched
  end

  @testset "5. the multiplier acts only on the interface: values at the bounds do not depend on λ" begin
    c = dw_cantilever(; nx = 40, ny = 20)
    m = dw(; substeps = 1)
    fo, z, s = rd_start(c, m)
    z0 = copy(z)                                               # the seed: values at 0 and 1 and an interface
    OPT.step!(m.step, s, z, fo, c.problem.ctx)                 # fills ψ and the sensitivity
    lb, ub = OPT.bounds(fo)
    R1 = RDM._dw_react!(similar(z), z0, s, 0.0, 0.25, lb, ub)
    R2 = RDM._dw_react!(similar(z), z0, s, 100.0, 0.25, lb, ub)
    at = findall(v -> v == 0.0 || v == 1.0, z0)
    mid = findall(v -> 0.05 < v < 0.95, z0)
    @test length(at) > 100 && !isempty(mid)
    @test R1[at] == R2[at]
    @test all(i -> R2[i] <= R1[i], mid) && count(i -> R2[i] < R1[i], mid) >= length(mid) ÷ 2
  end

  @testset "6. Δt_lim from Gershgorin (0.375 h²/(τℓ²) on squares), T = n Δt; triangles; the explicit scheme is stable" begin
    c = dw_cantilever(; nx = 40, ny = 20, frozen = false)
    fo, z, s = rd_start(c, dw())
    @test s.Δtlim ≈ 0.375 * (2 / 40)^2 / (DW_τ * 2) rtol = 1e-12
    # T = 25 in n = ⌈T/(4/3 Δt_lim)⌉ substeps: here T/(4/3 Δt_lim) = 8 up to rounding, n = 8 (not 9)
    @test s.nsub == s.substeps == 8 && s.Δt == 25 / 8
    @test RDM._dw_substeps(DoubleWell(), 3.75 * (1 - 4eps())) == 5 && RDM._dw_substeps(DoubleWell(), 3.75 * (1 - 1e-9)) == 6
    @test RDM._dw_substeps(DoubleWell(T = 1.0), 1e3) == 1 && RDM._dw_substeps(DoubleWell(substeps = 3), 1e-3) == 3
    for T in (5.0, 50.0)
      fo, z, s = rd_start(c, dw(; T))
      @test s.nsub == ceil(Int, T / (4 / 3 * s.Δtlim) - 1e-9) && s.Δt == T / s.nsub && s.Δt <= 4 / 3 * s.Δtlim * (1 + 1e-12)
    end
    fo, z, s = rd_start(c, dw(; scheme = :explicit))
    @test s.nsub == 32 && s.Δt == 25 / 32 && s.Δt <= s.Δtlim / 3
    # substeps given (T then has no effect: giving both is an error): Δt = :auto is cfl Δt_lim (the settings of Takezawa)
    @test_throws r"give T .* or substeps = n, not both" DoubleWell(T = 25.0, substeps = 20)
    fo, z, s = rd_start(c, dw(; scheme = :explicit, substeps = 20))
    @test s.Δt ≈ s.Δtlim / 3 rtol = 1e-14
    @test s.substeps == s.nsub == 20
    ct = dw_cantilever(; nx = 20, ny = 10, frozen = false, simplex = true)
    fo, z, st = rd_start(ct, dw())
    @test isfinite(st.Δtlim) && st.Δtlim > 0
    @test st.hmean ≈ sqrt(2 / 400) rtol = 1e-12                    # the mean cell size of the event
    @test s.hmean ≈ 2 / 40 rtol = 1e-12
    # T far too large for the mesh: a clear error (not an InexactError)
    @test_throws r"more than 1000000" RDM._dw_substeps(DoubleWell(T = 1e30), 1.0)
    @test_throws r"more than 1000000" solve!(dw_cantilever(; nx = 20, ny = 10).problem, dw(; T = 1e30); maxiter = 1, verbose = 0)
    # the explicit diffusion alone at cfl 1/3 and at the bound: values in [0, 1], energy zᵀKz not increasing
    for cfl in (1 / 3, 1.0)
      m = dw(; scheme = :explicit, cfl, substeps = 1)
      fo, z, s = rd_start(c, m)
      e = Float64[]
      for _ in 1:30
        RDM._dw_diffuse!(m.step, s, z)
        z .= s.φs
        push!(e, dot(z, s.K * z))
      end
      @test all(<=(1e-12 * e[1]), diff(e)) && all(v -> -1e-12 <= v <= 1 + 1e-12, z)
    end
    @test_throws r"cfl must be ≤ 1" DoubleWell(scheme = :explicit, cfl = 1.5)
    c = dw_cantilever(; nx = 20, ny = 10)
    @test_throws r"above the bound of stability" solve!(c.problem, ReactionDiffusion(; τ = DW_τ, Δt = 10.0,
                                                          reaction = DoubleWell(scheme = :explicit, substeps = 20)); maxiter = 1, verbose = 0)
    @test_throws r"with DoubleWell the substep is T/n.*substeps = n" ReactionDiffusion(; τ = DW_τ, Δt = 1.0, reaction = DoubleWell())
  end

  @testset "7. allocations and counts per iteration" begin
    c = dw_cantilever(; nx = 40, ny = 20)
    for scheme in (:implicit, :explicit)
      m = dw(; scheme)
      fo, z, s = rd_start(c, m)
      z0 = copy(z)
      stepalloc(st, s, z, fo, ctx, z0) = (copyto!(z, z0); @allocated OPT.step!(st, s, z, fo, ctx))
      OPT.step!(m.step, s, z, fo, c.problem.ctx)
      stepalloc(m.step, s, z, fo, c.problem.ctx, z0)
      ns, nf = s.nsolves, s.nfactor
      a = stepalloc(m.step, s, z, fo, c.problem.ctx, z0)
      @test s.nfactor == nf
      if scheme === :explicit
        @test a == 0 && s.nsolves == ns && s.nfactor == 0 && s.ns === nothing    # no linear system, no factorisation
      else
        @test s.nsolves == ns + s.nsub
        RDM._rd_solve!(s.φs, m.step, s, s.r)
        @test a <= s.nsub * @allocated RDM._rd_solve!(s.φs, m.step, s, s.r)
      end
      lb, ub = OPT.bounds(fo)
      aa, V = OPT.linear_constraint(fo, 1)
      G = RDM._DWLinearG(aa, V, z, s, lb, ub, 0.25, 0.0)
      G(0.1)
      @test (@allocated G(0.1)) == 0
      @test (@allocated RDM.grey_measure(m.step, s)) == 0
      # the events of an iteration allocate nothing unless there is an event
      evalloc(st, s, k, met, f) = @allocated RDM.iteration_events!(st, s, k, 100, met, f)
      evalloc(m.step, s, 1, true, 1.0)
      evalloc(m.step, s, 2, false, 1.0)
      @test evalloc(m.step, s, 3, true, 1.0) == 0 && evalloc(m.step, s, 4, false, 1.0) == 0
      @test (@allocated RDM.stop_allowed(m.step, s, 5)) == 0
    end
    r = solve!(dw_cantilever(; nx = 40, ny = 20).problem, dw(; settle = false); maxiter = 10, stop = nothing, verbose = 0)
    @test r.state.nsolves == 8 * 10 && r.state.nfactor == 1             # 8 substeps per iteration, one factorisation
    @test all(==(1), r.history.forward[2:end]) && all(==(0), r.history.adjoint)
  end

  @testset "8. cantilever 100×50, κ = 4e-5, T = 25 (the study of the mesh): 5 substeps, settles, stops after it" begin
    io = IOBuffer()
    c = cantilever_dw(; nx = 100, κ = 4e-5, verbose = 1, io)
    r = c.result
    out = String(take!(io))
    k = r.state.settle_iter
    @test r.state.nsub == 5 && r.state.Δt == 5.0
    @test occursin("double well: T = 25 per iteration = 5 substeps × Δt = 5 (Δt_lim = 3.75)", out)
    @test occursin("Δt = :auto = T/n = 25/n per substep (n = ⌈T/(cfl Δt_lim)⌉, cfl 1.33)", out)
    # the default stop waits for the iteration after the settling (with the windows met at the settling it stopped
    # there before: 66 on this case; now settles at 66 and stops at 67)
    @test r.status === :converged && r.reason === :all
    @test r.state.settled && r.state.substeps == 1 && 10 <= k < r.iterations
    @test occursin("double well settled at iteration $k (constraints met and objective range below 0.001 over 10 iterations)", out)
    @test occursin("(the default stop of the method is held until then; a stop given to solve! is not)", out)
    @test abs(c.volume - 0.5) <= 1e-3
    @test JCA100_LO <= r.objective <= JCA100_HI
  end

  @testset "8b. the default stop is held until the settling; a stop of the user is not" begin
    m = dw()
    fo, z, s = rd_start(dw_cantilever(; nx = 20, ny = 10), m)
    @test s.nsub == 2 && !RDM.stop_allowed(m.step, s, 5)
    s.settled, s.settle_iter, s.substeps = true, 5, 1
    @test !RDM.stop_allowed(m.step, s, 5) && RDM.stop_allowed(m.step, s, 6)
    m1 = dw(; settle = false)
    fo, z, s = rd_start(dw_cantilever(; nx = 20, ny = 10), m1)
    @test RDM.stop_allowed(m1.step, s, 1)
    m2 = dw(; T = 1.0)                                         # one substep from the start: nothing to wait for
    fo, z, s = rd_start(dw_cantilever(; nx = 20, ny = 10), m2)
    @test s.nsub == 1 && RDM.stop_allowed(m2.step, s, 1)
    @test RDM.stop_allowed(rd().step, nothing, 1)
    # a function of the user stops at the first iteration with the constraints met, long before the settling
    r = solve!(dw_cantilever(; nx = 20, ny = 10).problem, dw(); maxiter = 60, verbose = 0, stop = (info, h, ctx) -> true)
    @test r.status === :converged && r.reason === :custom && !r.state.settled
  end

  @testset "9. the bridge of Takezawa et al. (2010), 100×50, κ = 4e-5, :hex, 100 iterations: J against its reference" for method in (:default, :takezawa2010)
    b = bridge(; nx = 100, κ = 4e-5, method)
    @test b.result.iterations == 100
    lo, hi = method === :default ? (JBR100_LO, JBR100_HI) : (JBR100T_LO, JBR100T_HI)
    @test lo <= b.result.objective <= hi
    @test abs(b.result.history.constraints.volume[end]) <= 1e-4 * 2 / 0.6           # 1e-4 |D| in units of V_max = 0.6
    @test b.asymmetry_phase <= 0.05                     # symmetric about x = 1 up to a few cells
    @test 4 <= b.holes <= 10
  end

  @testset "9b. the default stop with the well: the bridge 100×50 settles and stops after the merge of holes" begin
    m = ReactionDiffusion(; τ = 2e-5, reaction = DoubleWell())
    @test RDM.default_stop(m.step).material_tol == 1e-2 && RDM.default_stop(rd().step).material_tol == 2e-3
    b = bridge(; nx = 100, κ = 4e-5, stop = :default, maxiter = 300)
    r = b.result
    # measured: settles at 105, stops at 116, J 23.263 (with the objective range alone it stopped before the merge of the
    # holes, J 24.8 on 200×100)
    @test r.status === :converged && r.reason === :all
    @test r.state.settled && r.state.settle_iter < r.iterations
    @test JBR100_LO <= r.objective <= JBR100_HI && 4 <= b.holes <= 10
    @test abs(r.history.constraints.volume[end]) <= 1e-4 * 2 / 0.6
    # shown in the banner; a stop of the user replaces it
    io = IOBuffer()
    solve!(dw_cantilever(; nx = 20, ny = 10).problem, dw(); maxiter = 1, verbose = 1, io)
    @test occursin("material change < 0.01 over 5 iterations, with the constraints met in the windows (default)", String(take!(io)))
    io = IOBuffer()
    solve!(dw_cantilever(; nx = 20, ny = 10).problem, dw(); maxiter = 1, verbose = 1, io,
           stop = StopCriteria(material_tol = 2e-3, objective_range = 1e-3, combine = :all))
    out = String(take!(io))
    @test occursin("material change < 0.002 over 5 iterations", out) && !occursin("(default) · maxiter", out)
  end

  @testset "9c. a first objective over 1000 times the final one does not stop the range of the objective early" begin
    # the seed of Takezawa et al. on 40 × 20 (holes over the supports: J₁ ≈ 1.2e5, J ≈ 28): the former floor
    # 1e-3 |J₁| of objective_range was met from iteration 43 (measured), before the range relative to the window
    b = bridge(; nx = 40, seed = :takezawa, stop = StopCriteria(objective_range = 1e-3), maxiter = 300)
    r = b.result
    h = r.history.objective
    steady(k, d) = (w = view(h, k-9:k); maximum(w) - minimum(w) <= 1e-3 * d(w))
    @test r.status === :converged && r.reason === :objective_range
    @test h[1] > 1000 * h[end]
    @test findfirst(k -> steady(k, w -> maximum(abs, w)), 10:length(h)) + 9 == r.iterations
    @test findfirst(k -> steady(k, w -> max(maximum(abs, w), 1e-3 * abs(h[1]))), 10:length(h)) + 9 < r.iterations - 10
  end

  @testset "10. the deprecated preset :takezawa2010, keywords that replace it, banner; TAKEZAWA2010 of validation" begin
    m = @test_deprecated ReactionDiffusion(:takezawa2010; τ = 5e-6)
    dwp = m.step.reaction
    @test dwp.W == 0.25 && dwp.η == 20.0 && dwp.scheme === :explicit && dwp.substeps == 20 && !dwp.settle
    @test m.step.volume.max_step == 0.01 && m.step.normalize isa RDM._TakezawaNorm
    @test RDM.method_label(m) == "ReactionDiffusion(:takezawa2010)"
    g, mm = [1.0, -2.0, 0.5], [0.1, 0.2, 0.3]
    @test m.step.normalize(g, mm, 2.0) ≈ 1 / sqrt(sum(g .^ 2 ./ mm)) rtol = 1e-14
    # the named tuple of validation/common/presets.jl: the same keywords, the same normalisation bit for bit
    mt = ReactionDiffusion(; τ = 5e-6, TAKEZAWA2010...)
    @test mt.step.reaction == dwp && mt.step.volume.max_step == 0.01 && mt.step.normalize(g, mm, 2.0) == m.step.normalize(g, mm, 2.0)
    rp = solve!(dw_cantilever(; nx = 20, ny = 10).problem, m; maxiter = 4, stop = nothing, verbose = 0)
    rt = solve!(dw_cantilever(; nx = 20, ny = 10).problem, mt; maxiter = 4, stop = nothing, verbose = 0)
    @test rp.history.objective == rt.history.objective && get_values(rp.z) == get_values(rt.z)
    @test m.step.sensitivity[] === :design && mt.step.sensitivity[] === :design      # deduced: the exact derivative
    m2 = @test_deprecated ReactionDiffusion(:takezawa2010; τ = 5e-6, reaction = DoubleWell())
    @test m2.step.reaction.scheme === :implicit && m2.sources[:reaction] === :given && m2.sources[:normalize] === :preset
    c = dw_cantilever(; nx = 20, ny = 10)
    io = IOBuffer()
    solve!(c.problem, m2; maxiter = 2, stop = nothing, verbose = 1, io)
    s = String(take!(io))
    @test occursin("reaction: DoubleWell(W = 0.25, η = 10, T = 25, cfl = :auto, scheme = :implicit, settle = true) (given, replaces :takezawa2010)", s)
    @test occursin("normalize: L² norm without |D| (Takezawa et al. 2010) (:takezawa2010)", s)
    # the defaults with the well: the exact sensitivity, L2Norm(), 1 % of |D| per iteration, the event of Δt
    io = IOBuffer()
    solve!(dw_cantilever(; nx = 20, ny = 10).problem, dw(); maxiter = 2, stop = nothing, verbose = 1, io)
    s = String(take!(io))
    @test occursin("sensitivity: exact (DoubleWell)", s) && occursin("L2Norm() (default)", s)
    # 20×10: Δt_lim = 0.375 h²/κ = 9.375, T/(4/3 Δt_lim) = 2 up to rounding: 2 substeps of 12.5 (the event after the
    # setup; the banner gives the formula and does not assemble K and m)
    @test occursin("Δt = :auto = T/n = 25/n per substep (n = ⌈T/(cfl Δt_lim)⌉, cfl 1.33)", s)
    @test occursin("target lowered 1 % of |D| per iteration", s)
    @test occursin("double well: T = 25 per iteration = 2 substeps × Δt = 12.5 (Δt_lim = 9.38), width of the interface " *
                   "ξ = √(τℓ²/2W) = 0.283 h̄ (h̄ = (|D|/cells)^(1/d) = 0.1, the mean cell size)", s)
    @test occursin("inside the tilt of the double well", s) && occursin("2 substeps of Δt = 12.5 per iteration (Δt_lim 9.38)", s)
    @test !occursin("settled", s)                              # 2 iterations: nothing settles (no fraction of maxiter)
    # substeps given: the banner and the event say so
    io = IOBuffer()
    solve!(dw_cantilever(; nx = 20, ny = 10).problem, dw(; substeps = 3); maxiter = 1, stop = nothing, verbose = 1, io)
    s = String(take!(io))
    @test occursin("Δt = :auto = 1.33 Δt_lim per substep", s) && occursin("= 3 substeps × Δt = 12.5 (substeps given)", s)
    @test sprint(show, DoubleWell()) == "DoubleWell(W = 0.25, η = 10, T = 25, cfl = :auto, scheme = :implicit, settle = true)"
    @test sprint(show, DoubleWell(substeps = 20, scheme = :explicit, cfl = 0.5)) ==
          "DoubleWell(W = 0.25, η = 10, substeps = 20, cfl = 0.5, scheme = :explicit, settle = true)"
  end

  @testset "11. errors" begin
    c = rd_cantilever(; nx = 20, ny = 10, init = x -> clamp(hypot(x[1] - 1, x[2] - 0.5) - 0.2, -1.0, 1.0))
    for iface in (Ersatz(), Step())
      ci = rd_cantilever(; nx = 20, ny = 10, interface = iface, init = x -> clamp(hypot(x[1] - 1, x[2] - 0.5) - 0.2, -1.0, 1.0))
      @test_throws r"double well with an interface map is not offered" solve!(ci.problem, dw(); maxiter = 1, verbose = 0)
    end
    @test_throws r":physical is not offered with DoubleWell" ReactionDiffusion(; τ = 1e-4, sensitivity = :physical, reaction = DoubleWell())
    @test (@test_deprecated ReactionDiffusion(; τ = 1e-4, sensitivity = :design, reaction = DoubleWell())).sensitivity === :design
    cs = rd_cantilever(; nx = 20, ny = 10, design = :density, init = 0.7)            # no interface
    @test_throws r"starting design has none.*hole_seed\(Vφ; bounds = \(0.0, 1.0\)" solve!(cs.problem, dw(); maxiter = 1, verbose = 0)
    @test_throws r"sensitivity = :design with an interface map is not a method.*DoubleWell" solve!(c.problem,
      (@test_deprecated ReactionDiffusion(; τ = 1e-4, sensitivity = :design)); maxiter = 1, verbose = 0)
    @test_throws r"W must be positive" DoubleWell(W = 0.0)
    @test_throws r"η must be positive" DoubleWell(η = -1.0)
    @test_throws r"substeps must be nothing or at least 1" DoubleWell(substeps = 0)
    @test_throws r"T must be positive" DoubleWell(T = 0.0)
    @test_throws r"T must be positive" DoubleWell(T = Inf)
    @test_throws r"scheme must be" DoubleWell(scheme = :kim)
    @test_throws r"cfl must be" DoubleWell(cfl = 0.0)
    @test_throws r"reaction must be nothing or DoubleWell" ReactionDiffusion(; τ = 1e-4, reaction = :well)
    cz = dw_cantilever(; nx = 20, ny = 10)
    @test_throws r"needs τ > 0" solve!(cz.problem, ReactionDiffusion(; τ = 0.0, reaction = DoubleWell()); maxiter = 1, verbose = 0)
    # τ = 0 with the substeps and Δt given: no width of the interface in the event (no NaN)
    io = IOBuffer()
    solve!(dw_cantilever(; nx = 20, ny = 10).problem, ReactionDiffusion(; τ = 0.0, Δt = 1.0, reaction = DoubleWell(substeps = 2));
           maxiter = 1, stop = nothing, verbose = 1, io)
    out = String(take!(io))
    @test occursin("no diffusion (τ = 0): no width of the interface", out) && !occursin("NaN h", out)
    # τ, κ or Δt continued (a Parameter) with the well: refused (n and Δt are set once from Δt_lim at the start)
    @test_throws r"τ must be a number, not a Parameter" ReactionDiffusion(; τ = Parameter(DW_τ), reaction = DoubleWell())
    @test_throws r"κ must be a number, not a Parameter" ReactionDiffusion(; κ = Parameter(2DW_τ), reaction = DoubleWell())
    @test_throws r"Δt must be a number, not a Parameter" ReactionDiffusion(; τ = DW_τ, Δt = Parameter(1.0),
                                                                          reaction = DoubleWell(substeps = 2))
    @test_throws r"τ must be a number, not a Parameter" ReactionDiffusion(; τ = Parameter(5e-6), TAKEZAWA2010...)
    @test ReactionDiffusion(; τ = Parameter(DW_τ)).step.τ isa Parameter   # without the well: as before
  end

  @testset "11b. κ = τℓ² instead of τ" begin
    @test_throws r"give τ or κ = τ ℓ².*not both" ReactionDiffusion(; τ = DW_τ, κ = 2DW_τ, reaction = DoubleWell())
    @test_throws r"τ is mandatory.*or give κ = τ ℓ² instead" ReactionDiffusion(; reaction = DoubleWell())
    @test_throws r"κ must be a number ≥ 0" ReactionDiffusion(; κ = -1.0, reaction = DoubleWell())
    c = dw_cantilever(; nx = 20, ny = 10)
    fo, z, s = rd_start(c, ReactionDiffusion(; κ = 2DW_τ, reaction = DoubleWell()))
    fo, z, s2 = rd_start(c, dw())
    @test s.τ ≈ DW_τ rtol = 1e-14
    @test s.Δtlim ≈ s2.Δtlim rtol = 1e-12
    @test s.nsub == s2.nsub
    fo, z, s3 = rd_start(rd_cantilever(; nx = 20, ny = 10), ReactionDiffusion(; κ = 4e-4))
    @test s3.τ ≈ 2e-4 rtol = 1e-14                                       # also without the well
    @test startswith(sprint(show, ReactionDiffusion(; κ = 1e-5, reaction = DoubleWell())), "ReactionDiffusion(κ = 1.0e-5")
    io = IOBuffer()
    solve!(dw_cantilever(; nx = 20, ny = 10).problem, ReactionDiffusion(; κ = 2DW_τ, reaction = DoubleWell()); maxiter = 1,
           stop = nothing, verbose = 1, io)
    @test occursin("κ = 0.0004 (τ = κ/ℓ² = 0.0002, ℓ² = 2)", String(take!(io)))
    io = IOBuffer()
    solve!(dw_cantilever(; nx = 20, ny = 10).problem, dw(); maxiter = 1, stop = nothing, verbose = 1, io)
    @test occursin("τ = 0.0002 (κ = τℓ² = 0.0004, ℓ² = 2)", String(take!(io)))
  end

  @testset "11c. κ and τ = κ/ℓ² give the same run bit for bit, with and without the well" begin
    # κ = 1e-5 (one substep on 40×20, no settling) and κ = 4e-4 (8 substeps, settles)
    for (κ, n) in ((1e-5, 1), (4e-4, 8))
      fo, z, s = rd_start(dw_cantilever(; nx = 40, ny = 20), ReactionDiffusion(; κ, reaction = DoubleWell()))
      @test s.c === κ                                # κ is the coefficient of the operator, without a round trip
      τ = κ / s.ℓ2                                   # ℓ² = |D|^(2/d) as the method computes it (the quadrature of |D|)
      @test τ * s.ℓ2 == κ                            # (on this mesh τ ℓ² gives κ back exactly)
      go(m) = solve!(dw_cantilever(; nx = 40, ny = 20).problem, m; maxiter = 200, verbose = 0)
      ra = go(ReactionDiffusion(; κ, reaction = DoubleWell()))
      rb = go(ReactionDiffusion(; τ, reaction = DoubleWell()))
      @test ra.state.c === rb.state.c === κ && ra.state.nsub == rb.state.nsub == n && ra.state.Δtlim == rb.state.Δtlim
      @test get_values(ra.z) == get_values(rb.z) && ra.history.objective == rb.history.objective
      @test ra.iterations == rb.iterations && ra.state.settle_iter == rb.state.settle_iter
      n > 1 && @test ra.state.settled
    end
    κ = 4e-4                                         # without the well
    fo, z, s = rd_start(rd_cantilever(; nx = 30, ny = 15), ReactionDiffusion(; κ))
    τ = κ / s.ℓ2
    @test s.c === κ && τ * s.ℓ2 == κ
    run0(m) = solve!(rd_cantilever(; nx = 30, ny = 15).problem, m; maxiter = 20, stop = nothing, verbose = 0)
    ra = run0(ReactionDiffusion(; κ))
    rb = run0(ReactionDiffusion(; τ))
    @test get_values(ra.z) == get_values(rb.z) && ra.history.objective == rb.history.objective
  end

  @testset "12. the grey measure and grey_tol; the settling" begin
    c = dw_cantilever(; nx = 40, ny = 20)
    m = dw()
    fo, z, s = rd_start(c, m)
    mv = s.m
    @test RDM.grey_measure(m.step, s) ≈ sum(4 .* mv .* z .* (1 .- z)) / 2.0 rtol = 1e-14
    # against the monitor Grey (Gridap's integral of the P1 field: the same order, not the same quadrature)
    set_values!(c.z, z)
    HyperShape.update_fields!(c.θ, c.problem.ctx)
    gg = HyperShape.Methods.grey_value(Grey(c.θ, c.dΩ.domain))
    @test abs(RDM.grey_measure(m.step, s) - gg) <= 0.25 * gg
    r = solve!(dw_cantilever(; nx = 40, ny = 20).problem, dw(); maxiter = 60, verbose = 0,
               stop = StopCriteria(grey_tol = 0.5))
    @test r.status === :converged && r.reason === :grey_tol && r.iterations < 60   # once the volume is met
    cd = rd_cantilever(; nx = 20, ny = 10, design = :density)
    @test_throws r"grey_tol needs" solve!(cd.problem, DensityMethod(OC()); maxiter = 2, verbose = 0, stop = StopCriteria(grey_tol = 0.01))
    @test HyperShape.Operators.num_solves(cd.S) == 0                       # refused at the setup, before any solve
    @test StopCriteria(grey_tol = 0.01).grey_tol == 0.01 && StopCriteria().grey_tol === nothing
    @test_throws ArgumentError StopCriteria(grey_tol = 0.0)
    # settle = false keeps the substeps (the settling of a run: test 8)
    r = solve!(dw_cantilever(; nx = 40, ny = 20).problem, dw(; settle = false); maxiter = 20, stop = nothing, verbose = 0)
    @test !r.state.settled && r.state.substeps == r.state.nsub == 8
    # the settling depends only on the objective: a short run does not settle
    r = solve!(dw_cantilever(; nx = 40, ny = 20).problem, dw(); maxiter = 8, stop = nothing, verbose = 0)
    @test !r.state.settled && r.state.substeps == 8
    # settle = (window, tol): the window and the tolerance of the settling
    io = IOBuffer()
    r = solve!(dw_cantilever(; nx = 40, ny = 20).problem, dw(; settle = (window = 5, tol = 1e-2)); maxiter = 60,
               stop = nothing, verbose = 1, io)
    out = String(take!(io))
    k = r.state.settle_iter
    f = r.history.objective
    @test r.state.settled && length(r.state.fwin) == 5 && k >= 5
    @test (maximum(f[k-4:k]) - minimum(f[k-4:k])) / maximum(abs, f[k-4:k]) <= 1e-2
    @test occursin("objective range below 0.01 over 5 iterations", out) && occursin("settle = (window = 5, tol = 0.01)", out)
    r10 = solve!(dw_cantilever(; nx = 40, ny = 20).problem, dw(); maxiter = k, stop = nothing, verbose = 0)
    @test !r10.state.settled                                   # the default window and tolerance: not yet at k
    @test DoubleWell(settle = (window = 20,)).settle_tol == 1e-3 && DoubleWell(settle = (tol = 1e-2,)).settle_window == 10
    @test DoubleWell(settle = true).settle && !DoubleWell(settle = false).settle
    @test_throws r"settle takes window and tol" DoubleWell(settle = (windows = 5,))
    @test_throws r"settle window must be an integer ≥ 2" DoubleWell(settle = (window = 1,))
    @test_throws r"settle tol must be a positive number" DoubleWell(settle = (tol = 0.0,))
  end

  @testset "13. checkpoint and resume bit for bit, across the settling, with a larger maxiter" begin
    c0 = dw_cantilever(; nx = 40, ny = 20)
    r0 = solve!(c0.problem, dw(); maxiter = 200, verbose = 0)
    @test r0.state.settled
    # the solves of the step: the substeps before the settling, 1 after it (no φ̌ with the well)
    @test r0.history.step_solves == [i <= r0.state.settle_iter ? r0.state.nsub : 1 for i in 1:r0.iterations]
    @test sum(r0.history.step_solves) == r0.state.nsolves
    for k in (r0.state.settle_iter - 3, r0.state.settle_iter)   # the window partly filled; settled at k (stop held)
      dir = mktempdir()
      c1 = dw_cantilever(; nx = 40, ny = 20)
      # a short run (maxiter = k), resumed with maxiter = 200: the uninterrupted run
      r1 = solve!(c1.problem, dw(); maxiter = k, verbose = 0, checkpoint = Checkpoint(filepath = dir, every = 1000))
      @test r1.iterations == k && r1.status === :maxiter
      c2 = dw_cantilever(; nx = 40, ny = 20)
      r2 = solve!(c2.problem, dw(); maxiter = 200, verbose = 0, resume = dir)
      @test r2.iterations == r0.iterations && r2.status === r0.status && r2.state.settle_iter == r0.state.settle_iter
      @test get_values(r2.z) == get_values(r0.z) && r2.history.objective == r0.history.objective
      @test r2.state.nsolves == count_substeps(r0, k)            # the substeps of the iterations after k
      @test r2.history.step_solves == r0.history.step_solves     # the history accumulates across the resume
      st = load_checkpoint(dir)
      @test keys(st.memory) == RDM._dw_memory_keys(10)
    end
    # the substeps follow the settling and the setup: a memory saved with other substeps (another T) is not stale
    m, m2 = dw(), dw(; T = 50.0)
    fo, z, s = rd_start(dw_cantilever(; nx = 20, ny = 10), m)
    fo2, z2, s2 = rd_start(dw_cantilever(; nx = 20, ny = 10), m2)
    @test s.nsub == 2 && s2.nsub == 4
    OPT.restore_memory!(m2.step, s2, z2, OPT.memory(m.step, s, z))
    @test s2.substeps == 4 && !s2.settled
    s.settled, s.substeps, s.settle_iter = true, 1, 7
    OPT.restore_memory!(m2.step, s2, z2, OPT.memory(m.step, s, z))
    @test s2.substeps == 1 && s2.settled && s2.settle_iter == 7
    m5 = dw(; settle = (window = 5,))
    fo5, z5, s5 = rd_start(dw_cantilever(; nx = 20, ny = 10), m5)
    @test_throws r"settle window of 5" OPT.restore_memory!(m5.step, s5, z5, OPT.memory(m.step, s, z))
  end

  @testset "15. a continuation jump starts the settling again" begin
    P = Parameter(1.0)                                         # continued but not used: only the jump matters
    io = IOBuffer()
    r = solve!(dw_cantilever(; nx = 40, ny = 20).problem, dw(); maxiter = 100, stop = nothing, verbose = 1, io,
               continuation = Continuation(pc = P => [1.0, 1.0]; every = 50))
    ks = [parse(Int, m.captures[1]) for m in eachmatch(r"double well settled at iteration (\d+)", String(take!(io)))]
    s = r.state
    @test length(ks) == 2 && ks[1] < 50 < ks[2] && s.settled && s.settle_iter == ks[2]
    # n substeps before each settling, 1 after it, n again from the jump (iteration 51)
    n = s.nsub
    @test s.nsolves == n * ks[1] + (50 - ks[1]) + n * (ks[2] - 50) + (100 - ks[2])
  end

  @testset "15b. checkpoint and resume across a continuation jump, before and after the second settling: bit for bit" begin
    P = Parameter(1.0)
    cont() = Continuation(pc = P => [1.0, 1.0]; every = 50)
    r0 = solve!(dw_cantilever(; nx = 40, ny = 20).problem, dw(); maxiter = 100, stop = nothing, verbose = 0,
                continuation = cont())
    k2 = r0.state.settle_iter
    @test r0.state.settled && 50 < k2 < 100
    for k in (40, 50, 51, 53, k2)        # before the jump, at it, just after it, in the second window, at the settling
      dir = mktempdir()
      r1 = solve!(dw_cantilever(; nx = 40, ny = 20).problem, dw(); maxiter = k, stop = nothing, verbose = 0,
                  continuation = cont(), checkpoint = Checkpoint(filepath = dir, every = 1000))
      @test r1.iterations == k
      r2 = solve!(dw_cantilever(; nx = 40, ny = 20).problem, dw(); maxiter = 100, stop = nothing, verbose = 0,
                  continuation = cont(), resume = dir)
      @test get_values(r2.z) == get_values(r0.z) && r2.history.objective == r0.history.objective
      @test r2.state.settled && r2.state.settle_iter == k2 && r2.iterations == r0.iterations
      @test r2.history.step_solves == r0.history.step_solves
    end
  end

  @testset "17. a rebuild of the discretisation (RebuildCheck): the run of FixedMesh bit for bit, the setup event once" begin
    io1, io2 = IOBuffer(), IOBuffer()
    r1 = solve!(dw_factories(), dw(); maxiter = 80, stop = nothing, verbose = 1, io = io1)
    r2 = solve!(dw_factories(), dw(); maxiter = 80, stop = nothing, verbose = 1, io = io2,
                discretization = RDM.RebuildCheck(every = 15))
    o1, o2 = String(take!(io1)), String(take!(io2))
    k = r1.state.settle_iter
    @test r1.state.settled && 15 < k < 75                    # rebuilds before and after the settling
    @test get_values(r1.z) == get_values(r2.z) && r1.history.objective == r2.history.objective
    @test r2.state.settled && r2.state.settle_iter == k && r2.state.substeps == 1
    @test r1.history.step_solves == r2.history.step_solves
    @test occursin(", step $(sum(r1.history.step_solves)) · time", o1) && occursin(", step $(sum(r2.history.step_solves)) · time", o2)
    @test count(r"iteration \d+: design and physics rebuilt", o2) == 5
    @test count("· double well: T =", o1) == count("· double well: T =", o2) == 1    # not repeated at a rebuild
    @test count("double well settled at iteration", o2) == 1 && !occursin("(resumed)", o2)
  end

  @testset "16. with the well: a rejected step, a nonlinear native constraint, consistent mass, a solve of the user" begin
    for scheme in (:implicit, :explicit)
      c = dw_cantilever(; nx = 20, ny = 10, solve_flag = ctx -> !(ctx.iteration == 3 && ctx.attempt == 1))
      r = solve!(c.problem, dw(; scheme); maxiter = 5, stop = nothing, verbose = 0)
      st = r.history.step
      @test r.iterations == 5 && st[3] == st[2] / 2 && st[4] == st[2]          # halved, then back
      # the target of the rejected iteration lowered in proportion (half of 1 % of |D|)
      g = r.history.constraints.volume
      @test abs(g[3] - (g[2] - 0.01)) <= VT && abs(g[4] - (g[3] - 0.02)) <= VT
      @test r.state.nfactor == (scheme === :implicit ? 3 : 0)                 # the start, the shrink and back
    end
    # a native constraint that is not linear: the root through the oracle
    c = dw_cantilever(; nx = 20, ny = 10, left_cap = 0.3)
    r = solve!(c.problem, ReactionDiffusion(; τ = DW_τ, reaction = DoubleWell(),
                                                lagrangian = AugmentedLagrangian(native = (:left,)));
               maxiter = 40, stop = nothing, verbose = 0)
    @test r.state.kind === :newton && r.history.constraints.left[end] <= 1e-4 * 2 / 0.6
    @test all(isfinite, r.history.objective) && all(v -> 0 <= v <= 1, get_values(r.z))
    # consistent mass, and a solve of the user (the same as LU)
    r = solve!(dw_cantilever(; nx = 20, ny = 10).problem, ReactionDiffusion(; τ = DW_τ, reaction = DoubleWell(), mass = :consistent);
               maxiter = 10, stop = nothing, verbose = 0)
    g = r.history.constraints.volume                            # the target lowered 1 % of |D| per iteration
    @test all(k -> abs(g[k] - (g[k-1] - 0.02)) <= VT, 2:length(g)) && all(isfinite, r.history.objective)
    calls = Ref(0)
    lusolve = (A, b, x, ctx, stale) -> (calls[] += 1; x .= A \ b)
    r1 = solve!(dw_cantilever(; nx = 20, ny = 10).problem, ReactionDiffusion(; τ = DW_τ, reaction = DoubleWell(), solve = lusolve);
                maxiter = 10, stop = nothing, verbose = 0)
    r2 = solve!(dw_cantilever(; nx = 20, ny = 10).problem, dw(); maxiter = 10, stop = nothing, verbose = 0)
    @test r1.history.objective ≈ r2.history.objective rtol = 1e-10
    @test calls[] == r1.state.nsolves == r2.state.nsolves
    @test sum(r1.history.step_solves) == calls[]                 # calls of the solve of the user (no φ̌ with the well)
  end

  @testset "14. P1 triangles and a small 3D case: they run, the volume is exact" begin
    c = dw_cantilever(; nx = 30, ny = 15, simplex = true)
    r = solve!(c.problem, dw(); maxiter = 70, stop = nothing, verbose = 0)
    @test abs(r.history.constraints.volume[end]) <= VT && all(isfinite, r.history.objective)
    @test all(v -> 0 <= v <= 1, get_values(r.z))
    model = CartesianDiscreteModel((0, 2, 0, 1, 0, 1), (16, 8, 8))
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    φ0 = hole_seed(V; bounds = (0.0, 1.0), radius = 0.2, spacing = 0.6)
    z = OptVariables(φ = FieldVariable(V; bounds = (0.0, 1.0), init = φ0))
    θ = PhysicalFields(DesignFields(z; φ = Identity()); χ = :φ => Identity())
    dΩ = Measure(Triangulation(model), 2)
    w = CellField(x -> 1 + x[1], Triangulation(model))
    F = Functional(θ -> sum(∫(w * (1 - θ.χ))dΩ), θ; dJdθ = θ -> (χ = δ -> ∫(-w * δ)dΩ,), scale = 1.0)
    p = Optimization(z; objective = F, constraints = (volume = Volume(θ, dΩ; max_fraction = 0.5),))
    r = solve!(p, ReactionDiffusion(; τ = DW_τ, reaction = DoubleWell()); maxiter = 60, stop = nothing, verbose = 0)
    g = r.history.constraints.volume
    @test r.state.ℓ2 ≈ 2.0^(2 / 3) && abs(g[end]) <= 1e-4 * 2 / 1.0 && all(isfinite, r.history.objective)
  end

end
