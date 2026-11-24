# hole_seed: starting designs with holes (patterns, margins, frozen zones, profile, connectivity, meshes, errors)
using Test
using Base.CoreLogging: with_logger, NullLogger      # Logging is not a test dependency
using Random
using Gridap
using Gridap.FESpaces
using HyperFEM
using HyperShape

const HSV = HyperShape.Variables
const SEED_MESHES = joinpath(@__DIR__, "..", "meshes")

# the 2 × 1 cantilever of the examples: clamp on the left, load on the middle fifth of the right side, cells under it
function seed_cantilever(; nx = 40, ny = 20, order = 1, simplex = false, frozen = true)
  model = CartesianDiscreteModel((0, 2, 0, 1), (nx, ny))
  simplex && (model = simplexify(model))
  labels = get_face_labeling(model)
  add_tag_from_vertex_filter!(labels, "clamp", model, x -> x[1] < 1e-9)
  add_tag_from_vertex_filter!(labels, "load", model, x -> x[1] > 2 - 1e-9 && abs(x[2] - 0.5) <= 0.1 + 1e-9)
  tag_adjacent_cells!(labels, "load_zone", model, "load")
  reffe = ReferenceFE(lagrangian, Float64, order)
  conf = order == 0 ? :L2 : :H1
  Vφ = frozen ? TestFESpace(model, reffe; conformity = conf, dirichlet_tags = "load_zone") :
       TestFESpace(model, reffe; conformity = conf)
  (; model, Vφ)
end

function dof_coords(V)
  X = [get_free_dof_values(interpolate(x -> x[k], V)) for k in 1:num_cell_dims(get_triangulation(V))]
  [Tuple(x[i] for x in X) for i in 1:num_free_dofs(V)]
end
dist(a, b) = sqrt(sum(abs2, a .- b))
mindist(c) = minimum(dist(c[i], c[j]) for i in eachindex(c) for j in i+1:length(c))
mindist(A, B) = isempty(A) || isempty(B) ? Inf : minimum(dist(a, b) for a in A for b in B)
tag_points(model, tag) = HSV._tag_vertices(model, (tag,), Val(num_cell_dims(model)))
function frozen_points(V)
  X, ids = get_cell_coordinates(get_triangulation(V)), get_cell_dof_ids(V)
  [Tuple(p) for c in eachindex(ids) if all(<(0), ids[c]) for p in X[c]]
end
quiet(f) = with_logger(f, NullLogger())
near(p, C; tol = 1e-9) = any(q -> dist(p, q) <= tol, C)
# the coefficient of variation of the measures of the Voronoi regions of the centres (cells by centroid)
function region_cv(V, C)
  G = HSV._seed_geometry(V)
  a = zeros(length(C))
  for (x, m) in zip(G.cen, G.meas)
    a[argmin([dist(x, p) for p in C])] += m
  end
  m = sum(a) / length(a)
  sqrt(sum(abs2, a .- m) / (length(a) - 1)) / m
end
nodeindex(xs, p) = findfirst(x -> dist(x, p) < 1e-9, xs)
meannn(c) = sum(minimum(dist(c[i], c[j]) for j in eachindex(c) if j != i) for i in eachindex(c)) / length(c)

@testset "hole_seed" begin
  @testset "values, bounds and FieldVariable (P$order, zone $frozen, $pattern)" for order in (0, 1), frozen in (true, false),
                                                                                       pattern in (:hex, :lattice, :cvt)
    c = seed_cantilever(; order, frozen)
    sp = pattern === :cvt ? 0.35 : 0.25
    φ = @test_logs hole_seed(c.Vφ; radius = 0.05, spacing = sp, pattern, exclude = "clamp")      # no warning
    @test φ isa Vector{Float64} && length(φ) == num_free_dofs(c.Vφ)
    @test all(v -> -1 <= v <= 1, φ) && any(<(0), φ) && any(==(1), φ)
    fv = FieldVariable(c.Vφ; bounds = (-1.0, 1.0), init = φ, frozen = frozen ? 1.0 : nothing)
    @test fv.init == φ
    φd = hole_seed(c.Vφ; radius = 0.05, spacing = sp, pattern, bounds = (0.0, 1.0))           # a density
    @test all(v -> 0 <= v <= 1, φd) && any(<(0.5), φd)
  end

  @testset "distances between centres" begin
    c = seed_cantilever(; frozen = false)
    for s in 1:3
      C = HSV.hole_centers(c.Vφ; radius = 0.05, spacing = 0.25, pattern = :poisson, rng = s)
      @test length(C) > 10 && mindist(C) >= 0.25 * (1 - 1e-9)
    end
    # :lattice: symmetric in the box, 0.125 + 0.25 i
    C = HSV.hole_centers(c.Vφ; radius = 0.05, spacing = 0.25, pattern = :lattice)
    @test length(C) == 32
    @test sort(unique(round.(first.(C), digits = 12))) ≈ 0.125 .+ 0.25 .* (0:7)
    @test sort(unique(round.(last.(C), digits = 12))) ≈ 0.125 .+ 0.25 .* (0:3)
    # :hex: a hole at the centre, nearest neighbours spacing apart, staggered rows √3/2 spacing apart, symmetric
    C = HSV.hole_centers(c.Vφ; radius = 0.05, spacing = 0.25)
    key(p) = round.(p, digits = 9)
    S = Set(key.(C))
    @test key((1.0, 0.5)) in S
    @test mindist(C) ≈ 0.25
    @test all(p -> key((2 - p[1], p[2])) in S && key((p[1], 1 - p[2])) in S, C)
    @test all(y -> any(k -> abs(y - (0.5 + k * sqrt(3) / 2 * 0.25)) < 1e-9, -3:3), last.(C))
    @test all(p -> any(k -> abs(p[1] - (1 + 0.125 * k)) < 1e-9, -16:16), C)
    # 3D :hex (face-centred cubic): 12 neighbours spacing apart around the centre
    m3 = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (12, 12, 12))
    V3 = TestFESpace(m3, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    C3 = HSV.hole_centers(V3; radius = 0.08, spacing = 0.32)
    @test mindist(C3) ≈ 0.32
    @test count(p -> abs(dist(p, (0.5, 0.5, 0.5)) - 0.32) < 1e-9, C3) == 12
    C3 = HSV.hole_centers(V3; radius = 0.08, spacing = 0.32, pattern = :lattice)
    @test mindist(C3) ≈ 0.32
  end

  @testset "jitter: holes never touch" begin
    c = seed_cantilever(; frozen = false)
    r, s = 0.05, 0.25
    for j in (0.5, 0.9)
      C = HSV.hole_centers(c.Vφ; radius = r, spacing = s, jitter = j, rng = 3)
      @test mindist(C) >= 2r + (1 - j) * (s - 2r) - 1e-12
      @test mindist(C) < s                                     # they moved
      C = HSV.hole_centers(c.Vφ; radius = r, spacing = s, pattern = :lattice, jitter = j, rng = 3)
      @test mindist(C) >= 2r + (1 - j) * (s - 2r) - 1e-12
    end
  end

  @testset "margins: excluded tags and frozen zones (P$order)" for order in (0, 1)
    c = seed_cantilever(; order)
    xs = dof_coords(c.Vφ)
    clamp_pts, zone_pts = tag_points(c.model, "clamp"), frozen_points(c.Vφ)
    @test !isempty(clamp_pts) && !isempty(zone_pts)
    for pattern in (:hex, :lattice, :cvt), margin in (0.05, 0.1)
      φ = quiet(() -> hole_seed(c.Vφ; radius = 0.05, spacing = 0.25, pattern, margin, exclude = "clamp"))
      void = xs[φ .< 0]
      @test mindist(void, clamp_pts) >= margin - 1e-12
      @test mindist(void, zone_pts) >= margin - 1e-12
      C = HSV.hole_centers(c.Vφ; radius = 0.05, spacing = 0.25, pattern, margin, exclude = "clamp")
      @test mindist(C, clamp_pts) >= 0.05 + margin - 1e-12 && mindist(C, zone_pts) >= 0.05 + margin - 1e-12
    end
  end

  @testset "margin = 0: holes reach the tag; frozen cells keep their neighbours solid" begin
    c = seed_cantilever()
    xs = dof_coords(c.Vφ)
    clamp_pts = tag_points(c.model, "clamp")
    kw = (radius = 0.06, spacing = 0.3, pattern = :lattice, origin = (0.075, 0.525), exclude = "clamp")
    φ = hole_seed(c.Vφ; margin = 0.0, kw...)
    @test mindist(xs[φ .< 0], clamp_pts) < 0.06
    φ = hole_seed(c.Vφ; kw...)
    @test mindist(xs[φ .< 0], clamp_pts) >= 0.06 - 1e-12
    # with margin ≥ √2 h every cell that shares a vertex with the zone has all its nodes in the solid (Ersatz() = 1)
    φ = hole_seed(c.Vφ; radius = 0.05, spacing = 0.25, margin = 0.1)
    ids = get_cell_dof_ids(c.Vφ)
    topo = Gridap.Geometry.get_grid_topology(c.model)
    c2v, v2c = Gridap.Geometry.get_faces(topo, 2, 0), Gridap.Geometry.get_faces(topo, 0, 2)
    near = Set(c2 for cl in eachindex(ids) if all(<(0), ids[cl]) for v in c2v[cl] for c2 in v2c[v])
    @test !isempty(near)
    @test all(cl -> all(d -> d < 0 || φ[d] >= 0, ids[cl]), near)
  end

  @testset "reproducibility" begin
    c = seed_cantilever()
    p(; kw...) = hole_seed(c.Vφ; radius = 0.05, spacing = 0.25, exclude = "clamp", kw...)
    q(; kw...) = p(; pattern = :cvt, spacing = 0.35, kw...)
    @test q() == q()                                                     # :cvt from :hex: deterministic
    @test q(rng = 1) == q(rng = 1) && q(rng = Xoshiro(7)) == q(rng = Xoshiro(7))
    @test q(rng = 1) != q(rng = 2) && q(rng = 1) != q()
    @test p() == p(rng = 5) == p(rng = Xoshiro(9))                       # :hex without jitter: no randomness
    @test p(pattern = :lattice) == p(pattern = :lattice, rng = 5)
    @test p(jitter = 0.5, rng = 4) == p(jitter = 0.5, rng = 4) != p(jitter = 0.5, rng = 5)
  end

  @testset "profile of the holes, bounds and base" begin
    c = seed_cantilever(; frozen = false)
    xs = dof_coords(c.Vφ)
    i0, i1, ih, i2 = (nodeindex(xs, p) for p in ((1.0, 0.5), (1.1, 0.5), (1.05, 0.5), (1.2, 0.5)))
    kw = (radius = 0.1, spacing = 0.5, pattern = :lattice, origin = (1.0, 0.5))
    φ = hole_seed(c.Vφ; kw...)
    @test φ[i0] == -1 && abs(φ[i1]) <= 1e-12 && φ[ih] ≈ -0.5 && φ[i2] == 1
    φ = hole_seed(c.Vφ; kw..., width = 0.05)
    @test φ[ih] ≈ -1 && abs(φ[i1]) <= 1e-12 && φ[nodeindex(xs, (1.15, 0.5))] ≈ 1
    φ = hole_seed(c.Vφ; kw..., bounds = (0.0, 1.0))
    @test φ[i0] == 0 && φ[i1] ≈ 0.5 && φ[ih] ≈ 0.25 && φ[i2] == 1
    @test maximum(hole_seed(c.Vφ; kw..., base = 0.5)) == 0.5
    @test maximum(hole_seed(c.Vφ; kw..., base = fill(0.25, length(xs)))) == 0.25
    φ = hole_seed(c.Vφ; kw..., base = x -> x[1] < 0.5 ? -1.0 : 1.0)
    @test all(i -> φ[i] == -1, findall(x -> x[1] < 0.5, xs)) && φ[i0] == -1 && φ[i2] == 1
    @test_throws DimensionMismatch hole_seed(c.Vφ; kw..., base = [0.0])
  end

  @testset "connectivity of the solid" begin
    c = seed_cantilever()
    kw = (radius = 0.05, spacing = 0.25, exclude = "clamp", connect = ("clamp", "load"))
    @test_logs hole_seed(c.Vφ; kw...)
    @test_logs hole_seed(c.Vφ; kw..., pattern = :cvt, spacing = 0.35)
    cut = x -> abs(x[1] - 1) < 0.06 ? -1.0 : 1.0                       # a void band across the beam
    @test_logs (:warn, r"does not join the tags clamp, load: the tag load is in another piece") hole_seed(c.Vφ; kw..., base = cut)
    @test_logs hole_seed(c.Vφ; kw..., base = cut, connect = nothing)
    @test_logs hole_seed(c.Vφ; kw..., base = cut, connect = ())
    # a void frozen zone: no solid at the load
    @test_logs (:warn, r"no solid cell at the tag load") hole_seed(c.Vφ; kw..., frozen = -1.0)
    # the default connect is exclude
    @test_logs (:warn, r"the tag load is in another piece") hole_seed(c.Vφ; radius = 0.05, spacing = 0.25,
                                                                       exclude = ("clamp", "load"), base = cut)
    # P0 and simplices
    c0 = seed_cantilever(; order = 0, simplex = true)
    @test_logs hole_seed(c0.Vφ; kw...)
    @test_logs (:warn, r"another piece") hole_seed(c0.Vφ; kw..., base = cut)
  end

  @testset "warnings: unresolved holes, thin solid; verbose" begin
    c = seed_cantilever(; nx = 20, ny = 10, frozen = false)
    @test_logs (:warn, r"thinner than two cells") (:warn, r"holes contain no value of the field") match_mode = :any hole_seed(c.Vφ;
      radius = 0.06, spacing = 0.25, pattern = :lattice, origin = (1.0, 0.5))
    @test_logs (:warn, r"thinner than two cells") hole_seed(c.Vφ; radius = 0.1, spacing = 0.35)
    @test_logs (:info, r"holes \(hex\), .* % of the values below the level") hole_seed(c.Vφ; radius = 0.1,
      spacing = 0.45, verbose = true)
  end

  @testset "errors" begin
    c = seed_cantilever()
    V = c.Vφ
    @test_throws r"bounds must be finite with lb < ub" hole_seed(V; radius = 0.05, bounds = (1.0, -1.0))
    @test_throws r"radius must be positive" hole_seed(V; radius = 0.0)
    @test_throws r"width must be positive" hole_seed(V; radius = 0.05, width = 0.0)
    @test_throws r"margin must be ≥ 0" hole_seed(V; radius = 0.05, margin = -0.1)
    @test_throws r"would overlap" hole_seed(V; radius = 0.05, spacing = 0.1)
    @test_throws r"pattern must be :hex, :lattice or :cvt" hole_seed(V; radius = 0.05, pattern = :cos)
    @test_throws r"pattern = :cvt with rng" hole_seed(V; radius = 0.05, pattern = :poisson)           # internal only
    @test_throws r"jitter must be in \[0, 1\)" hole_seed(V; radius = 0.05, jitter = 1.0)
    @test_throws r"jitter moves the holes of :hex and :lattice" hole_seed(V; radius = 0.05, pattern = :cvt, jitter = 0.5)
    @test_throws r"nholes and min_distance are options of pattern = :cvt" hole_seed(V; radius = 0.05, nholes = 10)
    @test_throws r"nholes and min_distance are options of pattern = :cvt" hole_seed(V; radius = 0.05, pattern = :lattice, min_distance = 0.2)
    @test_throws r"nholes must be a positive integer" hole_seed(V; radius = 0.05, pattern = :cvt, nholes = 0)
    @test_throws r"min_distance .* must be larger than 2 radius" hole_seed(V; radius = 0.05, pattern = :cvt, min_distance = 0.1)
    @test_throws r"rng must be nothing, an integer seed or an AbstractRNG" hole_seed(V; radius = 0.05, rng = 1.5)
    @test_throws r"each symmetry is a named tuple" hole_seed(V; radius = 0.05, symmetry = ((1.0, 0.5), (0.0, 1.0)))
    @test_throws r"needs normal = … \(a reflection\) or order" hole_seed(V; radius = 0.05, symmetry = (point = (1.0, 0.5),))
    @test_throws r"needs a point" hole_seed(V; radius = 0.05, symmetry = (normal = (0.0, 1.0),))
    @test_throws r"nonzero vector" hole_seed(V; radius = 0.05, symmetry = (point = (1.0, 0.5), normal = (0.0, 0.0)))
    @test_throws r"order of a rotation must be an integer ≥ 2" hole_seed(V; radius = 0.05, symmetry = (point = (1.0, 0.5), order = 1))
    @test_throws r"in 2D has no axis" hole_seed(V; radius = 0.05, symmetry = (point = (1.0, 0.5), axis = (0, 0, 1), order = 2))
    @test_throws r"not both" hole_seed(V; radius = 0.05, symmetry = (point = (1.0, 0.5), normal = (0, 1), order = 2))
    @test_throws r"do not generate a finite group" hole_seed(V; radius = 0.05, symmetry = ((point = (1.0, 0.5), normal = (0, 1)),
                                                                                              (point = (1.0, 0.5), normal = (sin(1), cos(1)))))
    @test_throws r"no tag named support \(in exclude\)" hole_seed(V; radius = 0.05, exclude = "support")
    @test_throws r"no tag named support \(in connect\)" hole_seed(V; radius = 0.05, connect = ("clamp", "support"))
    @test_throws r"origin must have 2 coordinates" hole_seed(V; radius = 0.05, origin = (1.0, 0.5, 0.0))
    @test_throws r"give the test space" hole_seed(TrialFESpace(V, 1.0); radius = 0.05)
    @test_throws r"no hole fits" hole_seed(V; radius = 0.05, exclude = "clamp", margin = 3.0)
    c = seed_cantilever(; nx = 10, ny = 5, frozen = false)          # h = 0.2: the holes fall between the nodes
    @test_throws r"no value of the field falls inside a hole" quiet(() -> hole_seed(c.Vφ; radius = 0.05, spacing = 0.4,
                                                                                   pattern = :lattice, origin = (0.1, 0.1)))
  end

  @testset "meshes: $name, P$order" for (name, mk, r) in (
      ("Cartesian quadrilaterals", () -> CartesianDiscreteModel((0, 1, 0, 1), (24, 24)), 0.08),
      ("Cartesian triangles", () -> simplexify(CartesianDiscreteModel((0, 1, 0, 1), (24, 24))), 0.08),
      ("Gmsh triangles", () -> DiscreteModelFromFile(joinpath(SEED_MESHES, "m2_s_16.json")), 0.1),
      ("Cartesian hexahedra", () -> CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (10, 10, 10)), 0.12),
      ("Cartesian tetrahedra", () -> simplexify(CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (6, 6, 6))), 0.15),
      ("Gmsh tetrahedra", () -> DiscreteModelFromFile(joinpath(SEED_MESHES, "m3_s_8.json")), 0.15)), order in (0, 1)
    model = mk()
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, order); conformity = order == 0 ? :L2 : :H1)
    for pattern in (:hex, :lattice, :cvt)
      φ = quiet(() -> hole_seed(V; radius = r, spacing = 3r, pattern))
      @test length(φ) == num_free_dofs(V) && all(v -> -1 <= v <= 1, φ) && any(<(0), φ) && any(>(0), φ)
      C = quiet(() -> HSV.hole_centers(V; radius = r, spacing = 3r, pattern))
      @test length(C) >= 2 && mindist(C) >= (pattern === :cvt ? 2.5r : 3r) * (1 - 1e-9)
      # every value below the level is inside a hole
      xs = dof_coords(V)
      @test all(x -> minimum(c -> dist(x, c), C) < r, xs[φ .< 0])
    end
  end

  @testset "void fraction of the patterns" begin
    model = CartesianDiscreteModel((0, 2, 0, 1), (80, 40))
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    r, s = 0.05, 0.25
    for (pattern, est) in ((:lattice, π * r^2 / s^2), (:hex, 2π * r^2 / (sqrt(3) * s^2)))
      z = OptVariables(φ = FieldVariable(V; bounds = (-1.0, 1.0), init = hole_seed(V; radius = r, spacing = s, pattern)))
      θ = PhysicalFields(DesignFields(z; φ = Identity()); χ = :φ => Ersatz())
      void = 1 - sum(∫(θ.χ)design_measures(θ; degree = 1).domain) / 2
      @test abs(void - est) <= 0.3 * est
      pattern === :lattice && @test abs(void - est) <= 0.05 * est           # 32 whole holes
    end
  end

  @testset ":cvt: distances, margins, more even than random points" begin
    c = seed_cantilever(; nx = 80, ny = 40)
    clamp_pts, zone_pts = tag_points(c.model, "clamp"), frozen_points(c.Vφ)
    r, s = 0.05, 0.2
    for rng in (nothing, 1, 2, 3)
      C = HSV.hole_centers(c.Vφ; radius = r, spacing = s, pattern = :cvt, rng, exclude = "clamp")
      @test length(C) > 20
      @test mindist(C) >= 2r + (s - 2r) / 2 - 1e-12
      @test mindist(C, clamp_pts) >= 2r - 1e-12 && mindist(C, zone_pts) >= 2r - 1e-12
      rng === nothing && continue
      P = HSV.hole_centers(c.Vφ; radius = r, spacing = s, pattern = :poisson, rng, exclude = "clamp")   # its start
      n0 = length(HSV.hole_centers(c.Vφ; radius = r, spacing = s, pattern = :cvt, exclude = "clamp"))
      @test abs(length(C) - n0) <= 0.1n0 + 2
      # measured (80 × 40, clamp and zone excluded): regions 0.16–0.22 against 0.26–0.31 for random points at least spacing apart
      @test region_cv(c.Vφ, C) < region_cv(c.Vφ, P)
    end
    # spacing is the mean distance to the nearest neighbour (the number of holes calibrated), also in a slender strip
    # and in a slender bar (3D), where the measure over the area per hole of :hex gives too many; nholes fixes the
    # number (one relaxation), min_distance the minimum
    # measured: 0.98–1.02 on the rectangle, 0.93 on the strip 1.5 spacing wide (documented)
    for (box, cells, n, tol) in (((0, 2, 0, 1), (80, 40), 2 / (sqrt(3) / 2 * s^2), (0.95, 1.05)),
                                 ((0, 4, 0, 0.3), (160, 12), 1.2 / (sqrt(3) / 2 * s^2), (0.9, 1.05)))
      Vb = TestFESpace(CartesianDiscreteModel(box, cells), ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
      for rng in (nothing, 1)
        C = HSV.hole_centers(Vb; radius = r, spacing = s, pattern = :cvt, rng)
        @test tol[1] <= meannn(C) / s <= tol[2]
        @test mindist(C) >= 2r + (s - 2r) / 2 - 1e-12
        @test all(p -> box[1] <= p[1] <= box[2] && box[3] <= p[2] <= box[4], C)
        C = HSV.hole_centers(Vb; radius = r, spacing = s, pattern = :cvt, rng, nholes = round(Int, n))
        @test length(C) == round(Int, n) && mindist(C) >= 2r + (s - 2r) / 2 - 1e-12
      end
    end
    Vb = TestFESpace(CartesianDiscreteModel((0, 2, 0, 1), (80, 40)), ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    C = HSV.hole_centers(Vb; radius = r, spacing = s, pattern = :cvt, min_distance = 0.18)
    @test mindist(C) >= 0.18 - 1e-12
    bar = CartesianDiscreteModel((0, 3, 0, 0.4, 0, 0.4), (45, 6, 6))           # 2 spacing thick
    Vb = TestFESpace(bar, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    C = HSV.hole_centers(Vb; radius = 0.06, spacing = 0.2, pattern = :cvt)
    # measured: 55 holes (the measure over s³/√2 gives 85), mean distance 0.97 spacing
    @test abs(meannn(C) / 0.2 - 1) <= 0.1 && mindist(C) >= 0.16 - 1e-12 && length(C) < 0.8 * 0.48 / (0.2^3 / sqrt(2))
    # an unstructured mesh: Gmsh tetrahedra
    V = TestFESpace(DiscreteModelFromFile(joinpath(SEED_MESHES, "m3_s_8.json")), ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    C = HSV.hole_centers(V; radius = 0.1, spacing = 0.35, pattern = :cvt)
    @test length(C) >= 4 && mindist(C) >= 0.2 + 0.075 - 1e-12
    @test all(p -> all(0 .<= p .<= 1), C)
  end

  @testset "symmetry: $pattern" for pattern in (:hex, :lattice, :cvt)
    c = seed_cantilever(; nx = 80, ny = 40)
    xs = dof_coords(c.Vφ)
    r, s = 0.05, pattern === :cvt ? 0.25 : 0.2
    dmin = pattern === :cvt ? 2r + (s - 2r) / 2 : s
    clamp_pts = tag_points(c.model, "clamp")
    mirror = (point = (1.0, 0.5), normal = (0.0, 1.0))                  # y = 0.5: the clamp and the load are symmetric
    C = @test_logs HSV.hole_centers(c.Vφ; radius = r, spacing = s, pattern, exclude = "clamp", symmetry = mirror)
    @test all(p -> near((p[1], 1 - p[2]), C), C)
    @test mindist(C) >= dmin - 1e-12 && mindist(C, clamp_pts) >= 2r - 1e-12
    φ = quiet(() -> hole_seed(c.Vφ; radius = r, spacing = s, pattern, exclude = "clamp", symmetry = mirror))
    @test all(i -> isapprox(φ[i], φ[nodeindex(xs, (xs[i][1], 1 - xs[i][2]))]; atol = 1e-12), eachindex(φ))
    # two planes (the load breaks x = 1: warning, and the seed is still symmetric)
    both = (mirror, (point = (1.0, 0.5), normal = (1.0, 0.0)))
    C = @test_logs (:warn, r"symmetry 2 does not map .* the frozen zones and excluded tags onto themselves") HSV.hole_centers(c.Vφ;
          radius = r, spacing = s, pattern, rng = pattern === :cvt ? 3 : nothing, exclude = "clamp", symmetry = both)
    @test all(p -> near((p[1], 1 - p[2]), C) && near((2 - p[1], p[2]), C), C)
    @test mindist(C) >= dmin - 1e-12 && mindist(C, clamp_pts) >= 2r - 1e-12
    # rotation of order 4 about the centre of a square (2D) and about the z axis (3D)
    sq = CartesianDiscreteModel((-1, 1, -1, 1), (40, 40))
    Vs = TestFESpace(sq, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    C = pattern === :hex ?                                               # the lattice of :hex has no 4-fold axis
        (@test_logs (:warn, r"does not map the lattice of :hex onto itself") HSV.hole_centers(Vs; radius = 0.08, spacing = 0.4,
                                                                                               pattern, symmetry = (point = (0, 0), order = 4))) :
        (@test_logs HSV.hole_centers(Vs; radius = 0.08, spacing = 0.4, pattern, symmetry = (point = (0, 0), order = 4)))
    @test all(p -> near((-p[2], p[1]), C), C) && mindist(C) >= (pattern === :cvt ? 0.16 + 0.12 : 0.4) - 1e-12
    bx = CartesianDiscreteModel((-1, 1, -1, 1, 0, 1), (12, 12, 6))
    Vb = TestFESpace(bx, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
    C = quiet(() -> HSV.hole_centers(Vb; radius = 0.12, spacing = 0.45, pattern, symmetry = (point = (0, 0, 0), axis = (0, 0, 1), order = 4)))
    @test length(C) >= 4 && all(p -> near((-p[2], p[1], p[3]), C), C)
    # an asymmetric mesh: a warning
    @test_logs (:warn, r"symmetry 1 does not map the mesh onto itself") match_mode = :any HSV.hole_centers(c.Vφ; radius = r,
      spacing = s, pattern, symmetry = (point = (0.7, 0.5), normal = (1.0, 0.0)))
  end

  @testset "symmetries of the lattice itself (3D, P0): no hole lost" begin
    m3 = CartesianDiscreteModel((-1, 1, -1, 1, -1, 1), (12, 12, 12))
    V3 = TestFESpace(m3, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
    C0 = HSV.hole_centers(V3; radius = 0.12, spacing = 0.5)                  # face-centred cubic, 44 holes
    @test length(C0) == 44
    for sym in ((point = (0, 0, 0), normal = (1, -1, 0)), (point = (0, 0, 0), normal = (1, 0, -1)),
                (point = (0, 0, 0), axis = (1, 1, 1), order = 3), (point = (0, 0, 0), normal = (1, 0, 0)),
                (point = (0, 0, 0), axis = (0, 0, 1), order = 4))
      C = @test_logs HSV.hole_centers(V3; radius = 0.12, spacing = 0.5, symmetry = sym)
      @test length(C) == 44 && all(p -> near(p, C0), C)                    # before: 34, 34, 41 (rounding at spacing)
    end
  end

  @testset "a symmetry the lattice does not have: warning" begin
    c = seed_cantilever(; nx = 80, ny = 40, frozen = false)
    @test_logs (:warn, r"does not map the lattice of :lattice onto itself") HSV.hole_centers(c.Vφ; radius = 0.05,
      spacing = 0.2, pattern = :lattice, origin = (1.03, 0.47), symmetry = (point = (1.0, 0.5), normal = (0.0, 1.0)))
    @test_logs HSV.hole_centers(c.Vφ; radius = 0.05, spacing = 0.2, pattern = :lattice,
                                symmetry = (point = (1.0, 0.5), normal = (0.0, 1.0)))   # the default lattice has it
  end

  @testset "design space on part of the model: the rest is solid for connect (P$order)" for order in (0, 1)
    c = seed_cantilever(; frozen = false)
    Ωd = Triangulation(c.model, [sum(xs)[1] / length(xs) < 1.5 for xs in get_cell_coordinates(Triangulation(c.model))])
    Vd = TestFESpace(Ωd, ReferenceFE(lagrangian, Float64, order); conformity = order == 0 ? :L2 : :H1)
    φ = @test_logs hole_seed(Vd; radius = 0.05, spacing = 0.25, exclude = "clamp", connect = ("clamp", "load"))
    @test length(φ) == num_free_dofs(Vd) && any(<(0), φ)
  end

  @testset "holes as large as the domain: error; mostly void: warning" begin
    c = seed_cantilever(; frozen = false)
    @test_throws r"every value of the field is inside a hole" hole_seed(c.Vφ; radius = 3.0)
    @test_logs (:warn, r"more holes than solid") match_mode = :any hole_seed(c.Vφ; radius = 0.3, spacing = 0.61)
    @test_logs (:warn, r"more holes than solid") match_mode = :any hole_seed(c.Vφ; radius = 0.3, spacing = 0.61, pattern = :cvt)
    # base below the level is not void from the holes: a uniform density 0.4 or level set -0.2 with holes (failed in 8bc0ff8)
    φ = @test_logs hole_seed(c.Vφ; radius = 0.05, spacing = 0.25, bounds = (0.0, 1.0), base = 0.4)
    @test all(v -> 0 <= v <= 0.4, φ) && any(==(0.0), φ)
    φ = @test_logs hole_seed(c.Vφ; radius = 0.05, spacing = 0.25, base = -0.2)
    @test all(v -> -1 <= v <= -0.2, φ) && any(==(-1.0), φ)
  end

  @testset ":cvt: nholes that do not fit, symmetry, one hole, min_distance above spacing" begin
    c = seed_cantilever(; nx = 80, ny = 40, frozen = false)
    @test_throws r"nholes = 5000 do not fit" HSV.hole_centers(c.Vφ; radius = 0.05, spacing = 0.2, pattern = :cvt, nholes = 5000)
    # close to the bound (≈ 102 at 0.15 in 2 × 1): placed, at the minimum, in a reasonable time
    t = @elapsed (C = HSV.hole_centers(c.Vφ; radius = 0.05, spacing = 0.2, pattern = :cvt, nholes = 100))
    @test length(C) == 100 && mindist(C) >= 0.15 - 1e-12 && t < 60            # 1300 took 48 s and 5000 did not end before
    sq = CartesianDiscreteModel((-1, 1, -1, 1), (40, 40))
    Vs = TestFESpace(sq, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    C = @test_logs (:warn, r"nholes = 10 asked, \d+ placed \(a symmetry places whole orbits\)") HSV.hole_centers(Vs;
          radius = 0.08, spacing = 0.4, pattern = :cvt, nholes = 10, symmetry = (point = (0, 0), order = 4))
    @test length(C) % 4 in (0, 1) && all(p -> near((-p[2], p[1]), C), C)
    # a domain of the size of spacing: one hole, not corrected (was 3, measured as "3 spacing" apart)
    Vo = TestFESpace(CartesianDiscreteModel((0, 0.2, 0, 0.2), (20, 20)), ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    @test length(HSV.hole_centers(Vo; radius = 0.05, spacing = 0.2, pattern = :cvt)) == 1
    @test_throws r"min_distance .* larger than spacing" hole_seed(c.Vφ; radius = 0.05, spacing = 0.2, pattern = :cvt, min_distance = 0.3)
    @test length(HSV.hole_centers(c.Vφ; radius = 0.05, spacing = 0.2, pattern = :cvt, min_distance = 0.3, nholes = 10)) == 10
  end

  @testset "symmetry internals: generic direction, centres onto axes" begin
    # a plane that contains the first generic direction (3, 2): another direction is chosen
    for sym in ((point = (0, 0), normal = (2.0, -3.0)), (point = (0, 0, 0), normal = (0.5345224838248488, -0.8017837257372732, 0.0)))
      D = length(sym.point)
      S = HSV._symmetry_group(sym, Val(D), 1.0)
      g = S.G[2]
      Aw = ntuple(i -> sum(g.A[i][j] * S.w[j] for j in 1:D), D)
      @test dist(Aw, S.w) >= 0.1
    end
    # near an axis of order 4 and near the crossing of two planes the centre goes onto the axis (the fixed set of the
    # subgroup), not to the mean of some of its images
    S = HSV._symmetry_group((point = (0, 0), order = 4), Val(2), 1.0)
    @test dist(HSV._snap(S, (0.01, 0.03), 0.1), (0.0, 0.0)) < 1e-12
    S = HSV._symmetry_group(((point = (0, 0), normal = (1, 0)), (point = (0, 0), normal = (0, 1))), Val(2), 1.0)
    @test dist(HSV._snap(S, (0.02, 0.03), 0.1), (0.0, 0.0)) < 1e-12
    @test dist(HSV._snap(S, (0.02, 0.5), 0.1), (0.0, 0.5)) < 1e-12            # near one plane only: onto it
    S = HSV._symmetry_group((point = (0, 0, 0), axis = (0, 0, 1), order = 4), Val(3), 1.0)
    @test dist(HSV._snap(S, (0.01, 0.02, 0.3), 0.1), (0.0, 0.0, 0.3)) < 1e-12
  end

  @testset "meshes: periodic, mixed, geometry of order 2" begin
    for (V, r) in ((TestFESpace(CartesianDiscreteModel((0, 1, 0, 1), (20, 20); isperiodic = (true, false)),
                                ReferenceFE(lagrangian, Float64, 1); conformity = :H1), 0.08),
                   (TestFESpace(CartesianDiscreteModel((0, 1, 0, 1), (20, 20); isperiodic = (true, true)),
                                ReferenceFE(lagrangian, Float64, 0); conformity = :L2), 0.08))
      φ = quiet(() -> hole_seed(V; radius = r, spacing = 3r))
      @test length(φ) == num_free_dofs(V) && any(<(0), φ) && any(>(0), φ)
    end
    n = 16
    id(i, j) = i + 1 + j * (n + 1)
    coords = [Point(i / n, j / n) for j in 0:n for i in 0:n]
    cells, types = Vector{Int}[], Int8[]
    for j in 0:n-1, i in 0:n-1                                   # quadrilaterals on the left, triangles on the right
      if i < n ÷ 2
        push!(cells, [id(i, j), id(i + 1, j), id(i, j + 1), id(i + 1, j + 1)]); push!(types, 1)
      else
        push!(cells, [id(i, j), id(i + 1, j), id(i, j + 1)]); push!(types, 2)
        push!(cells, [id(i + 1, j), id(i, j + 1), id(i + 1, j + 1)]); push!(types, 2)
      end
    end
    grid = Gridap.Geometry.UnstructuredGrid(coords, Gridap.Arrays.Table(cells), [Gridap.ReferenceFEs.QUAD4, Gridap.ReferenceFEs.TRI3], types)
    mm = Gridap.Geometry.UnstructuredDiscreteModel(grid)
    for order in (0, 1)
      V = TestFESpace(mm, ReferenceFE(lagrangian, Float64, order); conformity = order == 0 ? :L2 : :H1)
      φ = quiet(() -> hole_seed(V; radius = 0.08, spacing = 0.3, pattern = :cvt))
      @test length(φ) == num_free_dofs(V) && any(<(0), φ)
      G = HSV._seed_geometry(V)
      @test G.hmax ≈ sqrt(2) / n && minimum(G.h) ≈ 1 / n                   # triangles: hypotenuse; squares: side
    end
    # QUAD9 (geometry of order 2): the cell size from the vertices, 1/16 (was √2/16 from the 9 nodes)
    reffe = Gridap.ReferenceFEs.LagrangianRefFE(Float64, QUAD, 2)
    ξ = Gridap.ReferenceFEs.get_node_coordinates(reffe)
    id2(i, j) = i + 1 + j * (2n + 1)
    coords = [Point(i / (2n), j / (2n)) for j in 0:2n for i in 0:2n]
    cells = [[id2(2i + round(Int, 2p[1]), 2j + round(Int, 2p[2])) for p in ξ] for j in 0:n-1 for i in 0:n-1]
    m2 = Gridap.Geometry.UnstructuredDiscreteModel(Gridap.Geometry.UnstructuredGrid(coords, Gridap.Arrays.Table(cells), [reffe], fill(Int8(1), n^2)))
    V = TestFESpace(m2, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    G = HSV._seed_geometry(V)
    @test G.hmean ≈ 1 / n && all(c -> dist(c, (0.5, 0.5)) < 1, G.cen)
    @test any(<(0), @test_logs hole_seed(V; radius = 0.1, spacing = 0.36))     # no "thinner than two cells" warning (0.16 > 2/16)
  end

  @testset "cost: linear, no allocation per value" begin
    sp(n) = TestFESpace(CartesianDiscreteModel((0, 2, 0, 1), (2n, n)), ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    V1, V2 = sp(50), sp(100)
    for pattern in (:hex, :cvt)
      hole_seed(V1; radius = 0.05, spacing = 0.2, pattern); hole_seed(V2; radius = 0.05, spacing = 0.2, pattern)
      a1 = @allocated hole_seed(V1; radius = 0.05, spacing = 0.2, pattern)
      a2 = @allocated hole_seed(V2; radius = 0.05, spacing = 0.2, pattern)
      # 4 times the cells, the same holes: memory linear; measured :hex ≈ 240 B per value (the arrays and Gridap's
      # interpolation of the coordinates), was 4.8 kB
      @test a2 / a1 < 5
      @test (a2 - a1) / (num_free_dofs(V2) - num_free_dofs(V1)) < (pattern === :hex ? 600 : 3000)
    end
    # :cvt with rng and holes ∝ cells (radius 2.5 cells): the farthest-point fill is linear (memory ratio ≈ 4)
    hole_seed(V1; radius = 0.05, pattern = :cvt, rng = 1); hole_seed(V2; radius = 0.025, pattern = :cvt, rng = 1)
    a1 = @allocated hole_seed(V1; radius = 0.05, pattern = :cvt, rng = 1)
    a2 = @allocated hole_seed(V2; radius = 0.025, pattern = :cvt, rng = 1)
    @test a2 / a1 < 6
  end

  @testset "exported, no clash with Gridap and HyperFEM" begin
    @test Base.isexported(HyperShape, :hole_seed) && hole_seed === HyperShape.Variables.hole_seed
    @test !Base.isexported(HyperShape, :hole_centers)
  end
end
