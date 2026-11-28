# The problems of the validation cases (shared by the scripts of validation/ and by test/Validation; included by
# common.jl, not a library).
#
#   mbb(; case = :top88 | :simpl, …)  half MBB beam, SIMP, against top88 (Andreassen et al. 2011, data/top88)
#                                     and against SiMPL (Kim, Lazarov, Surowiec, Keith, arXiv:2411.19421)
#   cantilever3d(; …)                 3D cantilever (close to top3d), no published value: runs, Taylor tests, symmetry
#   bridge(; …)                       the bridge of Takezawa et al. (2010) with the double well (ReactionDiffusion +
#                                     DoubleWell): J = 23.3 published at 100 iterations
#   cantilever_dw(; …)                the cantilever 2 × 1 with the double well from hole_seed (mesh independence)
#
# Each returns a named tuple with the result of `solve!`, the cell means of the physical density and what the fixed
# report table needs (`cells`, `n`, `hcell`, `vf`, `jbin`: see report.jl and measures.jl).
# tag the vertices and edges whose vertices all satisfy `pred` (a roller on part of an edge)
function tag_faces!(model, name, pred)
  labels = get_face_labeling(model)
  topo = get_grid_topology(model)
  coords = Gridap.Geometry.get_vertex_coordinates(topo)
  ent = Gridap.Geometry.num_entities(labels) + 1
  for d in 0:1
    f2v = Gridap.Geometry.get_faces(topo, d, 0)
    for f in 1:length(f2v)
      all(v -> pred(coords[v]), f2v[f]) && (labels.d_to_dface_to_entity[d+1][f] = ent)
    end
  end
  add_tag!(labels, name, [ent])
end

# Helmholtz filter with radius rmin (r = rmin / (2√3), Lazarov and Sigmund 2011) and, with `cellmean`, the cell mean
# of the filtered field (the xPhys of top88)
function density_chain(dΩ, rmin; cellmean = false)
  filter = HelmholtzFilter(dΩ; rmin, ls = LUSolver())
  cellmean ? CellMean(dΩ) ∘ filter : filter
end

design_space(model, order) = order == 0 ? TestFESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2) :
                                          TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)

"""
    mbb(; case, nx, ny, order = 0, opt = OC(), vf, rmin, Emin, maxiter = 2000, stop = :default, cellmean = false,
        sensitivity = :exact, check = false, verbose = 0)

Half MBB beam with SIMP (p = 3, ν = 0.3), Helmholtz filter, no projection, volume of the physical density.
- `case = :top88`: (0, nx)×(0, ny), h = 1, plane stress, unit point load at the top-left node (the traction dual to the
  corner shape function on the first top edge), roller at the bottom-right node. With `cellmean = true` the
  formulation is that of top88 with the PDE filter.
- `case = :simpl`: (0, 3)×(0, 1), plane strain, body force (0, −1) in the disc of radius 0.05 at (0, 1), roller on the
  bottom edge for x > 3 − 1/32 (the MBB2 problem of the authors' code).
`sensitivity = :authors` replaces the exact derivative of the compliance by the one of the SiMPL code
(dohyun-cse/mfem, `topopt.cpp`), which halves the shear term: `λ tr(ε)² + μ ε:ε` instead of `λ tr(ε)² + 2μ ε:ε`.
`stop = :default` uses the default criteria of `solve!`. With `check = true` returns the report of
`check_consistency` instead of optimising.
"""
function mbb(; case, nx, ny, order = 0, opt = OC(), vf, rmin, Emin, maxiter = 2000, stop = :default, cellmean = false,
             sensitivity = :exact, check = false, verbose = 0)
  Lx, Ly = case === :top88 ? (Float64(nx), Float64(ny)) : (3.0, 1.0)
  h = Lx / nx
  model = CartesianDiscreteModel((0, Lx, 0, Ly), (nx, ny))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "sym", [1, 3, 7])
  add_tag_from_tags!(labels, "top", [3, 6])
  case === :top88 ? add_tag_from_tags!(labels, "roller", [2]) :
                    tag_faces!(model, "roller", x -> x[2] < 1e-9 && x[1] > 3 - 1 / 32 - 1e-9)
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  Γ = BoundaryTriangulation(model; tags = "top")
  dΓ = Measure(Γ, 2)
  Vρ = design_space(model, order)
  init = check ? [0.2 + 0.3 * abs(sin(2.3i)) for i in 1:num_free_dofs(Vρ)] : vf
  z, x, θ = single_field_design(:ρ, Vρ; bounds = (0.0, 1.0), init, map = density_chain(dΩ, rmin; cellmean))
  ν = 0.3
  λ = case === :top88 ? ν / (1 - ν^2) : ν / ((1 + ν) * (1 - 2ν))
  μ = 1 / (2(1 + ν))
  E(ρ) = Emin + ρ^3 * (1 - Emin)
  dE(ρ) = 3ρ^2 * (1 - Emin)
  σ(ε) = λ * tr(ε) * one(ε) + 2μ * ε
  sensitivity in (:exact, :authors) || throw(ArgumentError("mbb: sensitivity must be :exact or :authors"))
  σd = sensitivity === :exact ? σ : (ε -> λ * tr(ε) * one(ε) + μ * ε)     # :authors halves the shear term
  bc = BoundaryConditions(BoundaryCondition("sym", VectorValue(0.0, 0.0); mask = (true, false)),
                          BoundaryCondition("roller", VectorValue(0.0, 0.0); mask = (false, true)))
  V = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1), bc; conformity = :H1)
  U = TrialFESpace(V, bc, 1.0)
  a(u, v) = ∫((E ∘ θ.ρ) * (ε(v) ⊙ (σ ∘ ε(u))))dΩ
  # point load: the traction dual to the corner shape function on the first top edge (∫ t N₀ = 1, ∫ t N₁ = 0)
  tp(x) = VectorValue(0.0, x[1] < h ? -(4 - 6 * x[1] / h) / h : 0.0)
  fb(x) = VectorValue(0.0, x[1]^2 + (x[2] - 1)^2 < 0.05^2 ? -1.0 : 0.0)
  l = case === :top88 ? (v -> ∫(tp ⋅ v)dΓ) : (v -> ∫(fb ⋅ v)dΩ)
  J = case === :top88 ? ((u, θ) -> ∫(tp ⋅ u)dΓ) : ((u, θ) -> ∫(fb ⋅ u)dΩ)
  S = StateOperator(StaticLinearModel(l, a, U, V, bc; ls = LUSolver(), backend = :compiled), θ;
        adjoint = AdjointProblem(sensitivity = (u, p) -> (δ -> ∫((dE ∘ θ.ρ) * (ε(p) ⊙ (σd ∘ ε(u))) * δ)dΩ)))
  C = Functional(J, S; dJdu = SelfAdjoint(), scale = 1.0)
  Vol = Volume(θ, dΩ; max_fraction = vf)
  check && return check_consistency(C, Vol)
  problem = Optimization(z; objective = C, constraints = (volume = Vol,))
  method = DensityMethod(opt; metric = LumpedMass(dΩ))     # OC uses it only for the stationarity measure
  kw = stop === :default ? (;) : (; stop)
  t = @elapsed r = solve!(problem, method; maxiter, verbose, monitors = (grey = Grey(θ, dΩ),), kw...)
  ρ = cell_means(θ, dΩ, h * Ly / ny)
  form = (Eh, u, v) -> ∫(Eh * (ε(v) ⊙ (σ ∘ ε(u))))dΩ
  V0 = design_space(model, 0)
  jbin = binary_evaluator(form, l, U, V, V0; Emin)
  jdens = density_evaluator(form, l, U, V, V0; Emin)          # J of a design given by one density per cell
  (; result = r, time = t, ρ, nx, ny, cells = ρ, n = (nx, ny), hcell = (h, Ly / ny), vf, jbin, jdens)
end

"""
    top88_reference(nx, ny)

The reference of `data/top88` (top88.m of Andreassen et al. 2011 run in Octave, density filter ft = 2): the history
`c` printed by the code, its iterations and the final physical design `xPhys` as cell values in Gridap's order
(x fastest, y upwards). See `data/top88/README.md`.
"""
function top88_reference(nx, ny)
  dir = joinpath(@__DIR__, "..", "data", "top88")
  rows = [parse.(Float64, split(l, ",")) for l in readlines(joinpath(dir, "top88_$(nx)x$(ny)_hist.csv"))[2:end]]
  x = parse.(Float64, readlines(joinpath(dir, "top88_$(nx)x$(ny)_xPhys.csv")))
  # the array of top88 is nely × nelx, column-major, row 1 at the top
  (; c = [r[2] for r in rows], iterations = length(rows), xPhys = vec(permutedims(reverse(reshape(x, ny, nx), dims = 1))))
end

"""
    cantilever3d(; n = (24, 12, 12), order = 0, opt = OC(), vf = 0.3, maxiter = 300, stop = :default, check = false,
                 verbose = 0)

3D cantilever (0, 2)×(0, 1)×(0, 1), clamped at x = 0, downward traction on the bottom strip (z < h) of the face x = 2
(a strip of height h, close to top3d); SIMP p = 3, Emin = 1e-9, ν = 0.3, Helmholtz filter with rmin = 2.5h, no projection, volume fraction `vf`.
"""
function cantilever3d(; n = (24, 12, 12), order = 0, opt = OC(), vf = 0.3, maxiter = 300, stop = :default,
                      check = false, verbose = 0)
  model = CartesianDiscreteModel((0, 2, 0, 1, 0, 1), n)
  h = 2 / n[1]
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "clamp", [1, 3, 5, 7, 13, 15, 17, 19, 25])
  add_tag_from_tags!(labels, "load", [26])
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  Γ = BoundaryTriangulation(model; tags = "load")
  dΓ = Measure(Γ, 2)
  Vρ = design_space(model, order)
  init = check ? [0.2 + 0.3 * abs(sin(2.3i)) for i in 1:num_free_dofs(Vρ)] : vf
  z, x, θ = single_field_design(:ρ, Vρ; bounds = (0.0, 1.0), init, map = density_chain(dΩ, 2.5h))
  Emin, ν = 1e-9, 0.3
  λ, μ = ν / ((1 + ν) * (1 - 2ν)), 1 / (2(1 + ν))
  E(ρ) = Emin + ρ^3 * (1 - Emin)
  dE(ρ) = 3ρ^2 * (1 - Emin)
  σ(ε) = λ * tr(ε) * one(ε) + 2μ * ε
  t(x) = VectorValue(0.0, 0.0, x[3] < h ? -1.0 : 0.0)
  bc = BoundaryConditions(BoundaryCondition("clamp", VectorValue(0.0, 0.0, 0.0)))
  V = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), bc; conformity = :H1)
  U = TrialFESpace(V, bc, 1.0)
  a(u, v) = ∫((E ∘ θ.ρ) * (ε(v) ⊙ (σ ∘ ε(u))))dΩ
  l(v) = ∫(t ⋅ v)dΓ
  S = StateOperator(StaticLinearModel(l, a, U, V, bc; ls = LUSolver(), backend = :compiled), θ;
        adjoint = AdjointProblem(sensitivity = (u, p) -> (δ -> ∫((dE ∘ θ.ρ) * (ε(p) ⊙ (σ ∘ ε(u))) * δ)dΩ)))
  C = Functional((u, θ) -> ∫(t ⋅ u)dΓ, S; dJdu = SelfAdjoint(), scale = 1.0)
  Vol = Volume(θ, dΩ; max_fraction = vf)
  check && return check_consistency(C, Vol)
  problem = Optimization(z; objective = C, constraints = (volume = Vol,))
  method = DensityMethod(opt; metric = LumpedMass(dΩ))     # OC uses it only for the stationarity measure
  kw = stop === :default ? (;) : (; stop)
  tt = @elapsed r = solve!(problem, method; maxiter, verbose, monitors = (grey = Grey(θ, dΩ),), kw...)
  ρ = reshape(cell_means(θ, dΩ, h^3), n)
  jbin = binary_evaluator((Eh, u, v) -> ∫(Eh * (ε(v) ⊙ (σ ∘ ε(u))))dΩ, l, U, V, design_space(model, 0); Emin)
  (; result = r, time = tt, ρ, asymmetry = maximum(abs, ρ .- reverse(ρ, dims = 2)), dofs = num_free_dofs(U),
     cells = vec(ρ), n, hcell = (h, h, h), vf, jbin)
end

# The seed of Takezawa et al. (2010) for the bridge (their Fig. 9a): rows of rectangular holes 0.2 × 0.1, every 1/6 in
# height, 0.4 apart, alternate rows shifted by 0.2 (values 0 in the holes, 1 elsewhere)
function takezawa_bridge_seed(V)
  f(x) = begin
    k = round(Int, x[2] * 6)
    abs(x[2] - k / 6) <= 0.05 + 1e-9 || return 1.0
    xs = isodd(k) ? x[1] - 0.2 : x[1]
    abs(xs - 0.4 * round(xs / 0.4)) <= 0.1 + 1e-9 ? 0.0 : 1.0
  end
  collect(get_free_dof_values(interpolate(f, V)))
end

"""
    bridge(; nx = 200, ny = nx ÷ 2, seed = :hex, method = :default, κ = 1e-5, maxiter = 100,
           stop = nothing, snapshots = (), verbose = 0)

The bridge of Takezawa, Nishiwaki and Kitamura (2010, Section 5.2): domain 2 × 1, roller at (0, 0), pin at (2, 0),
unit point load downwards at (1, 0), plane stress (E = 1, ν = 0.3), volume fraction 0.3, `k(φ) = max(φ³, 1e-5)`, a
density φ ∈ [0, 1] with the double well. Published: J = 23.3 at 100 iterations with κ = 1e-5 on their 200 × 100 grid.
- `seed`: `:hex` (`hole_seed(…; radius = 0.1, spacing = 0.4)`, the default pattern) or `:takezawa` (their rectangles,
  `takezawa_bridge_seed`: only for the comparison with the published 23.3 on 200 × 100).
- `method`: `:default` (`ReactionDiffusion(; τ, reaction = DoubleWell())`, scheme (c)) or `:takezawa2010` (the
  settings of the paper, `ReactionDiffusion(; τ, TAKEZAWA2010...)`, presets.jl).
- `κ`: the diffusion `τ ℓ²` (`τ = κ/|D|`), a physical parameter (the same on every mesh). With a point load the
  compliance grows like `log(1/h)`: values on different meshes are not comparable.
`snapshots`: iterations whose cell means are kept (`snaps[k]`); other keywords go to the method (`ReactionDiffusion`),
`solve_kwargs` to `solve!`. Returns the result, the cell means, the interior holes,
the asymmetry `maxᵢ |χ − mirror(χ)|` of the cell means about x = 1, the fraction of cells whose phase (χ ≥ ½) differs
from the mirror one (`asymmetry_phase`; rounding breaks the symmetry and the evolution amplifies it: the designs are
symmetric up to a few cells, not to rounding) and the time.
"""
function bridge(; nx = 200, ny = nx ÷ 2, seed = :hex, method = :default, κ = 1e-5, maxiter = 100,
                stop = nothing, snapshots = (), verbose = 0, solve_kwargs = (;), kwargs...)
  h = 2 / nx
  model = CartesianDiscreteModel((0, 2, 0, 1), (nx, ny))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "roller", [1])
  add_tag_from_tags!(labels, "pin", [2])
  add_tag_from_tags!(labels, "bottom", [1, 2, 5])
  add_tag_from_vertex_filter!(labels, "load", model, x -> abs(x[1] - 1) < 1e-9 && x[2] < 1e-9)
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 4)
  dΓ = Measure(BoundaryTriangulation(model; tags = "bottom"), 2)
  bc = BoundaryConditions(BoundaryCondition("roller", VectorValue(0.0, 0.0); mask = (false, true)),
                          BoundaryCondition("pin", VectorValue(0.0, 0.0)))
  V = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1), bc; conformity = :H1)
  U = TrialFESpace(V, bc, 1.0)
  Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
  tags = ("roller", "pin", "load")
  φ0 = seed === :takezawa ? takezawa_bridge_seed(Vφ) :
       seed === :hex ? hole_seed(Vφ; bounds = (0.0, 1.0), radius = 0.1, spacing = 0.4, exclude = tags, connect = tags,
                                 width = 2h) : throw(ArgumentError("bridge: seed must be :takezawa or :hex"))
  z = OptVariables(φ = FieldVariable(Vφ; bounds = (0.0, 1.0), init = φ0))
  x = DesignFields(z)
  θ = PhysicalFields(x; χ = :φ)
  ν = 0.3
  λ, μ = ν / (1 - ν^2), 1 / (2(1 + ν))
  σ(e) = λ * tr(e) * one(e) + 2μ * e
  k(φ) = max(φ^3, 1e-5)
  dk(φ) = φ^3 > 1e-5 ? 3φ^2 : 0.0
  # unit point load at (1, 0) as in Takezawa et al. (it reproduces their J = 23.3 on 200 × 100, so it is kept here;
  # examples/bridge_dw.jl spreads the load over a fixed width instead): the traction dual to the shape function of
  # that node on the edge [1, 1 + h]
  t(x) = VectorValue(0.0, (1 - 1e-12 <= x[1] <= 1 + h + 1e-12) ? -(4 - 6 * (x[1] - 1) / h) / h : 0.0)
  a(u, v) = ∫((k ∘ θ.χ) * (ε(v) ⊙ (σ ∘ ε(u))))dΩ
  S = StateOperator(StaticLinearModel(v -> ∫(t ⋅ v)dΓ, a, U, V, bc; ls = LUSolver(), backend = :compiled), θ;
        adjoint = AdjointProblem(sensitivity = (u, p) -> (δ -> ∫((dk ∘ θ.χ) * (ε(p) ⊙ (σ ∘ ε(u))) * δ)dΩ)))
  C = Functional((u, θ) -> ∫(t ⋅ u)dΓ, S; dJdu = SelfAdjoint(), scale = 1.0)
  Vol = Volume(θ, dΩ; max_fraction = 0.3)
  problem = Optimization(z; objective = C, constraints = (volume = Vol,))
  τ = κ / 2                  # ℓ² = |D| = 2 (τ, not κ = …: the references were measured with this τ, bit for bit)
  m = method === :default ? ReactionDiffusion(; τ, merge((; reaction = DoubleWell()), kwargs)...) :
      method === :takezawa2010 ? ReactionDiffusion(; τ, TAKEZAWA2010..., kwargs...) :
      throw(ArgumentError("bridge: method must be :default or :takezawa2010"))
  V0 = FESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
  cells() = (HyperShape.update_fields!(θ, problem.ctx); collect(assemble_vector(v -> ∫(θ.χ * v)dΩ, V0)) ./ (h * h))
  snaps = Dict{Int,Vector{Float64}}()
  0 in snapshots && (snaps[0] = cells())
  mon = (snap = _Snapshots(Set{Int}(snapshots), snaps, cells, Ref(0)),)
  kw = stop === :default ? (;) : (; stop)
  tt = @elapsed r = solve!(problem, m; maxiter, verbose, monitors = isempty(snapshots) ? (;) : mon, kw..., solve_kwargs...)
  χ = cells()
  M = reshape(χ, nx, ny)
  P = M .>= 0.5
  jbin = binary_evaluator((Eh, u, v) -> ∫(Eh * (ε(v) ⊙ (σ ∘ ε(u))))dΩ, v -> ∫(t ⋅ v)dΓ, U, V, V0; Emin = 1e-5)
  (; result = r, χ, holes = interior_holes(χ, nx, ny), asymmetry = maximum(abs, M .- reverse(M, dims = 1)),
     asymmetry_phase = count(P .!= reverse(P, dims = 1)) / length(P), volume = sum(χ) / (nx * ny), time = tt, snaps, nx, ny,
     cells = χ, n = (nx, ny), hcell = (h, 1 / ny), vf = 0.3, jbin)
end

"""
    cantilever_dw(; nx = 200, ny = nx ÷ 2, κ = 1e-5, maxiter = 300, stop = :default, verbose = 0, io = stdout, kwargs...)

The cantilever 2 × 1 with the double well: clamped at x = 0, a unit traction downwards on x = 2, |y − ½| ≤ 0.1,
plane stress (E = 1, ν = 0.3), volume fraction 0.5, `k(φ) = max(φ³, 1e-5)`, a density φ ∈ [0, 1] from
`hole_seed(…; radius = 0.1, spacing = 0.4)` (`:hex`) and `ReactionDiffusion(; τ = κ/|D|, reaction = DoubleWell(; kwargs...))`
(`T = 25` per iteration by default). `κ` is physical: the case of the study of mesh independence
(`validation/mesh_double_well.jl`). Returns the result, the cell means, the interior holes, the volume and the time.
"""
function cantilever_dw(; nx = 200, ny = nx ÷ 2, κ = 1e-5, maxiter = 300, stop = :default, verbose = 0, io = stdout, kwargs...)
  h = 2 / nx
  model = CartesianDiscreteModel((0, 2, 0, 1), (nx, ny))
  labels = get_face_labeling(model)
  add_tag_from_vertex_filter!(labels, "clamp", model, x -> x[1] < 1e-9)
  add_tag_from_vertex_filter!(labels, "load", model, x -> x[1] > 2 - 1e-9 && abs(x[2] - 0.5) <= 0.1 + 1e-9)
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 4)
  dΓ = Measure(BoundaryTriangulation(model; tags = "load"), 2)
  bc = BoundaryConditions(BoundaryCondition("clamp", VectorValue(0.0, 0.0)))
  V = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1), bc; conformity = :H1)
  U = TrialFESpace(V, bc, 1.0)
  Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
  tags = ("clamp", "load")
  φ0 = hole_seed(Vφ; bounds = (0.0, 1.0), radius = 0.1, spacing = 0.4, exclude = tags, connect = tags, width = 2h)
  z = OptVariables(φ = FieldVariable(Vφ; bounds = (0.0, 1.0), init = φ0))
  θ = PhysicalFields(DesignFields(z); χ = :φ)
  ν = 0.3
  λ, μ = ν / (1 - ν^2), 1 / (2(1 + ν))
  σ(e) = λ * tr(e) * one(e) + 2μ * e
  k(φ) = max(φ^3, 1e-5)
  dk(φ) = φ^3 > 1e-5 ? 3φ^2 : 0.0
  t = VectorValue(0.0, -1.0)
  a(u, v) = ∫((k ∘ θ.χ) * (ε(v) ⊙ (σ ∘ ε(u))))dΩ
  S = StateOperator(StaticLinearModel(v -> ∫(t ⋅ v)dΓ, a, U, V, bc; ls = LUSolver(), backend = :compiled), θ;
        adjoint = AdjointProblem(sensitivity = (u, p) -> (δ -> ∫((dk ∘ θ.χ) * (ε(p) ⊙ (σ ∘ ε(u))) * δ)dΩ)))
  C = Functional((u, θ) -> ∫(t ⋅ u)dΓ, S; dJdu = SelfAdjoint(), scale = 1.0)
  problem = Optimization(z; objective = C, constraints = (volume = Volume(θ, dΩ; max_fraction = 0.5),))
  m = ReactionDiffusion(; τ = κ / 2, reaction = DoubleWell(; kwargs...))          # ℓ² = |D| = 2
  kw = stop === :default ? (;) : (; stop)
  tt = @elapsed r = solve!(problem, m; maxiter, verbose, io, kw...)
  V0 = FESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
  HyperShape.update_fields!(θ, problem.ctx)
  χ = collect(assemble_vector(v -> ∫(θ.χ * v)dΩ, V0)) ./ (h * h)
  jbin = binary_evaluator((Eh, u, v) -> ∫(Eh * (ε(v) ⊙ (σ ∘ ε(u))))dΩ, v -> ∫(t ⋅ v)dΓ, U, V, V0; Emin = 1e-5)
  (; result = r, χ, holes = interior_holes(χ, nx, ny), volume = sum(χ) / (nx * ny), time = tt, nx, ny,
     cells = χ, n = (nx, ny), hcell = (h, 1 / ny), vf = 0.5, jbin)
end

# a monitor that keeps the cell means at the iterations asked
struct _Snapshots{F}
  at::Set{Int}
  snaps::Dict{Int,Vector{Float64}}
  cells::F
  k::Base.RefValue{Int}
end
function HyperShape.Methods.monitor_value(s::_Snapshots)
  s.k[] += 1
  s.k[] in s.at && (s.snaps[s.k[]] = s.cells())
  0.0
end
