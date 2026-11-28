# Fast validation cases (the long ones are in validation/, outside CI): linear elasticity, the MBB beam of top88
# (the compliances published in Andreassen et al. 2011 and the design of top88.m run in Octave, validation/data/top88);
# Taylor tests of elasticity in 2D and 3D; a small 3D cantilever; the helpers of validation/common and run_all.jl.
using Test
isdefined(@__MODULE__, :report_row) || include(joinpath(@__DIR__, "..", "..", "validation", "common", "common.jl"))
isdefined(@__MODULE__, :run_all) || include(joinpath(@__DIR__, "..", "..", "validation", "run_all.jl"))

@testset "Validation" begin

  @testset "top88 in Octave (data/top88): the finite elements of the MBB problem" begin
    ref = top88_reference(60, 20)
    @test ref.iterations == 144 && length(ref.xPhys) == 1200
    @test sum(ref.xPhys) / 1200 ≈ 0.5 atol = 1e-5
    c = mbb(; case = :top88, nx = 60, ny = 20, vf = 0.5, rmin = 2.4, Emin = 1e-9, maxiter = 1, stop = nothing)
    # the final design of top88 evaluated here: 233.7132006834 with the finite elements of top88 (measured with a
    # plain-Julia transcription of them, 2026-11-27); top88 prints 233.71457 for the design before its last update
    @test c.jdens(ref.xPhys) ≈ 233.7132006834 rtol = 1e-9
    @test c.jdens(ref.xPhys) ≈ ref.c[end] rtol = 1e-4
    # the run of Octave reproduces Fig. 3 of Andreassen et al. (2011), density filter: 233.71 and 235.73
    @test round(ref.c[end], digits = 2) == 233.71
    ref150 = top88_reference(150, 50)
    @test ref150.iterations == 362 && length(ref150.xPhys) == 7500 && round(ref150.c[end], digits = 2) == 235.73
  end

  base = (; case = :top88, nx = 60, ny = 20, vf = 0.5, rmin = 2.4, Emin = 1e-9)
  c88 = 233.7132006834                                       # the final design of top88.m (ft = 2), see above
  t88 = (; opt = OC(adapt = nothing), stop = StopCriteria(change_tol = 0.01))

  # 237.60: the compliance of top88 with the PDE density filter of section 4.2 on 60×20, published in Fig. 4 of
  # Andreassen et al. (2011): HyperShape's formulation with the cell mean is that one (1 % rule; measured +0.00 %)
  @testset "MBB 60×20 against top88 with the PDE filter (237.60)" begin
    r = mbb(; base..., t88..., cellmean = true)              # the formulation of top88: cell mean of the filtered density
    @test r.result.status === :converged && r.result.iterations <= 150
    @test r.result.objective ≈ 237.60 rtol = 1e-3
    @test sum(r.ρ) / length(r.ρ) ≈ 0.5 atol = 1e-4
    # the cell means of the design as a P0 density give the same compliance (the formulation of top88)
    @test r.jdens(r.ρ) ≈ r.result.objective rtol = 1e-8
    @test r.result.objective ≈ c88 rtol = 0.02              # top88.m with its convolution filter: +1.67 % (the radius equivalence)
    @test r.jbin(r.ρ) ≈ r.jbin(top88_reference(60, 20).xPhys) rtol = 0.05     # the rule between methods, on J_bin
    for order in (0, 1)                                       # density at the Gauss points (the usual use of HyperShape)
      g = mbb(; base..., t88..., order)
      @test g.result.status === :converged
      @test g.result.objective ≈ 237.60 rtol = 2e-2
    end
  end

  @testset "Default optimisers ($(nameof(typeof(opt))), P$order)" for opt in (OC(), SiMPL()), order in (0, 1)
    r = mbb(; base..., opt, order, maxiter = 400)
    @test r.result.status === :converged
    @test r.result.objective ≈ 237.60 rtol = 2e-2
  end

  @testset "Taylor tests of elasticity (P$order)" for order in (0, 1)
    @test mbb(; base..., nx = 12, ny = 4, rmin = 1.5, order, check = true).passed
    @test mbb(; base..., nx = 12, ny = 4, rmin = 1.5, order, check = true, cellmean = true).passed
    @test cantilever3d(; n = (8, 4, 4), order, check = true).passed
  end

  @testset "Taylor test of the SiMPL setup (plane strain, body force, partial roller)" begin
    @test mbb(; case = :simpl, nx = 48, ny = 16, vf = 0.3, rmin = 0.1, Emin = 1e-6, check = true).passed
  end

  @testset "SiMPL with the authors' sensitivity: first iterate of their code (192×64)" begin
    # log SiMPL-B-MBB2-6-0.csv of dohyun-cse/mfem (rmin 0.05): S₀ 6.96277500e-2, step 1.25394956, F₁ 1.24728893e-2
    r = mbb(; case = :simpl, nx = 192, ny = 64, vf = 0.3, rmin = 0.05, Emin = 1e-6, opt = SiMPL(), sensitivity = :authors,
            maxiter = 1, stop = nothing)
    h = r.result.history
    @test h.stationarity[1] ≈ 6.962775e-2 rtol = 1e-6
    @test h.step[1] ≈ 1.2539496 rtol = 1e-6
    @test h.objective[1] ≈ 1.2472889e-2 rtol = 1e-6
    @test_throws ArgumentError mbb(; case = :simpl, nx = 12, ny = 4, vf = 0.3, rmin = 0.1, Emin = 1e-6, sensitivity = :half)
  end

  @testset "Helpers of validation/common: measures and report" begin
    # J_bin: a bar (0, 2)×(0, 1) clamped at x = 0, unit traction on x = 2, ν = 0: uniaxial and exact with Q1, so the
    # compliance of cells in series along x is Σ Lᵢ/Eᵢ (all solid: 2; left half solid, right half void: 1 + 1/Emin)
    model = CartesianDiscreteModel((0, 2, 0, 1), (4, 2))
    labels = get_face_labeling(model)
    add_tag_from_vertex_filter!(labels, "left", model, x -> x[1] < 1e-9)
    add_tag_from_vertex_filter!(labels, "right", model, x -> x[1] > 2 - 1e-9)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2)
    dΓ = Measure(BoundaryTriangulation(model; tags = "right"), 2)
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1); conformity = :H1, dirichlet_tags = "left")
    Uu = TrialFESpace(Vu, VectorValue(0.0, 0.0))
    σ0(e) = e                                                # E = 1, ν = 0: λ = 0, 2μ = 1
    jbin = binary_evaluator((E, u, v) -> ∫(E * (ε(v) ⊙ (σ0 ∘ ε(u))))dΩ, v -> ∫(VectorValue(1.0, 0.0) ⋅ v)dΓ, Uu, Vu,
                            design_space(model, 0); Emin = 1e-3)
    @test jbin(ones(8)) ≈ 2 rtol = 1e-10
    @test jbin(fill(0.5, 8)) ≈ 2 rtol = 1e-10                 # χ ≥ ½ is solid
    left = [i <= 2 ? 0.9 : 0.49 for j in 1:2 for i in 1:4]   # x fastest
    @test jbin(left) ≈ 1 + 1e3 rtol = 1e-10
    @test jbin(left .>= 0.5) ≈ 1 + 1e3 rtol = 1e-10           # a Bool per cell
    # binary volume, grey, perimeter
    @test binary_volume([0.2, 0.5, 0.7, 1.0]) == 0.75
    @test grey([0.0, 1.0, 0.5, 0.5]) ≈ 0.5
    one_cell = [i == 2 && j == 2 ? 1.0 : 0.0 for j in 1:3 for i in 1:3]
    @test perimeter(one_cell, (3, 3), (0.5, 1.0)) ≈ 2 * 1.0 + 2 * 0.5
    @test perimeter(vec([i == j == k == 2 ? 1.0 : 0.0 for i in 1:3, j in 1:3, k in 1:3]), (3, 3, 3), (1.0, 1.0, 1.0)) ≈ 6
    @test perimeter(ones(9), (3, 3), (1.0, 1.0)) == 0
    # interior holes: by faces, not touching the boundary, at least min_cells cells
    S = ones(7, 5)
    S[2, 2] = 0                                              # a hole of one cell
    S[4:5, 3] .= 0                                           # a hole of two cells
    S[7, 4] = 0                                              # void on the boundary: not a hole
    S[3, 3] = 0.3; S[3, 2] = 1                               # (3, 3) joins the two-cell hole by a face
    @test interior_holes(S) == 2
    @test interior_holes(vec(S), 7, 5) == 2
    @test interior_holes(S; min_cells = 2) == 1
    @test interior_holes(S .>= 0.5) == 2                     # a Bool design, true = solid
    D = ones(5, 5); D[2, 2] = 0; D[3, 3] = 0                 # touching by a corner only: two holes
    @test interior_holes(D) == 2
    C = ones(3, 3, 3); C[2, 2, 2] = 0                        # a closed cavity in 3D
    @test interior_holes(C) == 1
    C[2, 2, 3] = 0                                           # opened to the boundary
    @test interior_holes(C) == 0
    # the fixed report table
    c = mbb(; case = :top88, nx = 12, ny = 4, vf = 0.5, rmin = 1.5, Emin = 1e-9, maxiter = 5, stop = nothing)
    r = report_row("MBB 12×4", "P0, OC()", c; ref = 1.1 * c.result.objective)
    @test r.vs_ref ≈ 1 / 1.1 - 1 && r.basis === :J
    @test r.V ≈ 0.5 atol = 1e-3                             # the bisection of OC after 5 iterations
    @test r.forward >= 5 && r.adjoint == 0 && r.filter >= 5 && r.step === missing
    @test r.J_bin > 0 && 0 <= r.grey <= 1 && r.iterations == 5 && r.time_per_forward ≈ r.time / r.forward
    q = reference_row("MBB 12×4", "reference"; J = 1.0)
    @test q.J_bin === missing && q.holes === missing
    # a reference with its design: the measures of its cells; a given J_bin and ref recompute vs_ref
    q2 = reference_row("MBB 12×4", "design"; J = 2.0, cells = c.cells, n = c.n, hcell = c.hcell, jbin = c.jbin)
    @test q2.J_bin ≈ c.jbin(c.cells) && q2.V_bin == binary_volume(c.cells) && q2.holes == interior_holes(reshape(c.cells, c.n))
    r2 = report_row("MBB 12×4", "P0, OC()", c; ref = 2.0, basis = :J_bin, J_bin = 1.5)
    @test r2.vs_ref ≈ -0.25
    @test occursin("-25.00 % (J_bin)", markdown_table([r2]))
    dir = mktempdir()
    write_report("t", [q, r]; dir, io = devnull)
    lines = readlines(joinpath(dir, "t.csv"))
    @test length(lines) == 3 && startswith(lines[1], join(string.(REPORT_COLUMNS), ","))
    md = read(joinpath(dir, "t.md"), String)
    @test occursin("| case | variant | J | J_bin |", md) && occursin("| MBB 12×4 | P0, OC() |", md)
    # density_evaluator on the bar: a uniform density ρ gives E = Emin + (1 − Emin) ρ³ and J = 2/E
    jd = density_evaluator((E, u, v) -> ∫(E * (ε(v) ⊙ (σ0 ∘ ε(u))))dΩ, v -> ∫(VectorValue(1.0, 0.0) ⋅ v)dΓ, Uu, Vu,
                           design_space(model, 0); Emin = 1e-3)
    @test jd(fill(0.5, 8)) ≈ 2 / (1e-3 + (1 - 1e-3) * 0.125) rtol = 1e-10
    # the SVG figure: two designs and two curves
    f = joinpath(dir, "f.svg")
    design_figure(f; designs = [(trues(3, 2), ("a", "b")), (falses(3, 2), ("c", "d"))],
                  curves = [(; x = 1:3, y = [3.0, 2.0, 1.0], color = "black", dash = nothing, label = "one"),
                            (; x = 1:3, y = [1.0, 1.0, 1.0], color = "blue", dash = "5,3", label = "two")],
                  vlines = [(2, "black")], ylabel = "J", xlabel = "it", xmax = 3, xticks = (1, 3), ystep = 1, stack = true)
    svg = read(f, String)
    @test startswith(svg, "<svg") && endswith(svg, "</svg>\n") && count("<polyline", svg) == 2 && count("fill=\"black\"/>", svg) == 2
  end

  @testset "run_all.jl: a case that succeeds and one whose check fails" begin
    scripts, out = mktempdir(), mktempdir()
    write(joinpath(scripts, "good.jl"), "foreach(i -> println(\"line \", i), 1:50); @warn \"to stderr\"; println(\"end\")\n" *
          "write(joinpath(ENV[\"HYPERSHAPE_RESULTS\"], \"good.csv\"), \"case\\nx\\n\"); write(joinpath(ENV[\"HYPERSHAPE_RESULTS\"], \"good.md\"), \"table\\n\")\n")
    write(joinpath(scripts, "bad.jl"), "println(\"checks\"); exit(1)\n")
    failed = run_all(["good", "bad"]; dir = out, scripts, project = scripts, known = ["good", "bad"], io = devnull)
    @test failed == ["bad"]
    log = readlines(joinpath(out, "good.log"))
    @test all("line $i" in log for i in 1:50) && "end" in log && any(occursin("to stderr", l) for l in log)   # nothing overwritten
    @test readlines(joinpath(out, "report.csv")) == ["case", "x"] && occursin("Failed: bad", read(joinpath(out, "report.md"), String))
    @test_throws ArgumentError run_all(["good", "nope"]; dir = out, scripts, project = scripts, known = ["good", "bad"], io = devnull)
  end

  @testset "3D cantilever 8×4×4" begin
    r = cantilever3d(; n = (8, 4, 4), opt = SiMPL(), maxiter = 10, stop = nothing)
    h = r.result.history
    @test r.result.iterations == 10 && all(i -> h.objective[i+1] <= h.objective[i] * (1 + 1e-12), 1:9)
    @test all(<=(1e-6), h.constraints.volume) && r.asymmetry < 1e-8
  end
end
