# Validation cases of HyperShape (shared by the scripts of this folder and by test/Validation).
#
#   mbb(; case = :top88 | :simpl, …)  half MBB beam, SIMP, against top88 with the PDE filter (Andreassen et al. 2011)
#                                     and against SiMPL (Kim, Lazarov, Surowiec, Keith, arXiv:2411.19421)
#   cantilever3d(; …)                 3D cantilever (close to top3d), no published value: runs, Taylor tests, symmetry
#
# Both return a named tuple with the result of `solve!` and the cell means of the physical density.
using Gridap, Gridap.Geometry, HyperFEM, HyperShape, Printf

include(joinpath(@__DIR__, "top88.jl"))

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
# of the filtered field (the xPhys of top88), written as a second "filter" with the P0 mass matrix: an L2 projection
# onto P0. A named map for the cell mean is planned (CellMean, issue #4).
function density_chain(model, dΩ, rmin; cellmean = false)
  Vf = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
  r2 = (rmin / (2sqrt(3)))^2
  Sf = LinearOperator(StaticLinearModel((u, v) -> ∫(r2 * ∇(u) ⋅ ∇(v) + u * v)dΩ, TrialFESpace(Vf), Vf, NothingBC(); ls = LUSolver()))
  chain = HelmholtzFilter(Sf; measure = dΩ)
  if cellmean
    V0 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
    S0 = LinearOperator(StaticLinearModel((u, v) -> ∫(u * v)dΩ, TrialFESpace(V0), V0, NothingBC(); ls = LUSolver()))
    chain = HelmholtzFilter(S0; measure = dΩ) ∘ chain
  end
  chain
end

design_space(model, order) = order == 0 ? TestFESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2) :
                                          TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)

# cell means of the physical density
cell_means(θ, dΩ, vol) = collect(Gridap.get_array(∫(θ.ρ)dΩ)) ./ vol

"""
    mbb(; case, nx, ny, order = 0, opt = OC(), vf, rmin, Emin, maxiter = 2000, stop = :default, cellmean = false,
        check = false, verbose = 0)

Half MBB beam with SIMP (p = 3, ν = 0.3), Helmholtz filter, no projection, volume of the physical density.
- `case = :top88`: (0, nx)×(0, ny), h = 1, plane stress, unit point load at the top-left node (the traction dual to the
  corner shape function on the first top edge), roller at the bottom-right node. With `cellmean = true` the
  formulation is that of top88 with the PDE filter.
- `case = :simpl`: (0, 3)×(0, 1), plane strain, body force (0, −1) in the disc of radius 0.05 at (0, 1), roller on the
  bottom edge for x > 3 − 1/32 (the MBB2 problem of the authors' code).
`stop = :default` uses the default criteria of `solve!`. With `check = true` returns the report of
`check_consistency` instead of optimising.
"""
function mbb(; case, nx, ny, order = 0, opt = OC(), vf, rmin, Emin, maxiter = 2000, stop = :default, cellmean = false,
             check = false, verbose = 0)
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
  z, x, θ = single_field_design(:ρ, Vρ; bounds = (0.0, 1.0), init, physical = density_chain(model, dΩ, rmin; cellmean))
  ν = 0.3
  λ = case === :top88 ? ν / (1 - ν^2) : ν / ((1 + ν) * (1 - 2ν))
  μ = 1 / (2(1 + ν))
  E(ρ) = Emin + ρ^3 * (1 - Emin)
  dE(ρ) = 3ρ^2 * (1 - Emin)
  σ(ε) = λ * tr(ε) * one(ε) + 2μ * ε
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
  S = StateOperator(StaticLinearModel(l, a, U, V, bc; ls = LUSolver(), backend = :compiled), θ)
  C = Functional(J, S; adjoint = :self, scale = 1.0,
                 derivative = (u, w, θ) -> (δ -> ∫(-(dE ∘ θ.ρ) * (ε(w) ⊙ (σ ∘ ε(u))) * δ)dΩ))
  Vol = Volume(θ, dΩ; max_fraction = vf)
  check && return check_consistency(C, Vol)
  problem = Optimization(z; objective = C, constraints = Vol)
  method = opt isa OC ? SIMP(opt) : SIMP(opt; metric = LumpedMass(dΩ))
  kw = stop === :default ? (;) : (; stop)
  t = @elapsed r = solve!(problem, method; maxiter, verbose, grey = Grey(θ, dΩ), kw...)
  (; result = r, time = t, ρ = cell_means(θ, dΩ, h * Ly / ny), nx, ny)
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
  z, x, θ = single_field_design(:ρ, Vρ; bounds = (0.0, 1.0), init, physical = density_chain(model, dΩ, 2.5h))
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
  S = StateOperator(StaticLinearModel(l, a, U, V, bc; ls = LUSolver(), backend = :compiled), θ)
  C = Functional((u, θ) -> ∫(t ⋅ u)dΓ, S; adjoint = :self, scale = 1.0,
                 derivative = (u, w, θ) -> (δ -> ∫(-(dE ∘ θ.ρ) * (ε(w) ⊙ (σ ∘ ε(u))) * δ)dΩ))
  Vol = Volume(θ, dΩ; max_fraction = vf)
  check && return check_consistency(C, Vol)
  problem = Optimization(z; objective = C, constraints = Vol)
  method = opt isa OC ? SIMP(opt) : SIMP(opt; metric = LumpedMass(dΩ))
  kw = stop === :default ? (;) : (; stop)
  tt = @elapsed r = solve!(problem, method; maxiter, verbose, grey = Grey(θ, dΩ), kw...)
  ρ = reshape(cell_means(θ, dΩ, h^3), n)
  (; result = r, time = tt, ρ, asymmetry = maximum(abs, ρ .- reverse(ρ, dims = 2)), dofs = num_free_dofs(U))
end

# one line of a results table
row(name, r, ref = NaN) = @sprintf("| %s | %.6g | %s | %d | %s | %.1f s |", name, r.result.objective,
                                   isnan(ref) ? "—" : @sprintf("%+.2f %%", 100 * (r.result.objective / ref - 1)),
                                   r.result.iterations, r.result.status, r.time)
