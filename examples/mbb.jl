# Half MBB beam: compliance minimisation with a density method (SIMP interpolation, p = 3), a Helmholtz filter, a
# tanh projection with continuation, a volume constraint on the physical density, SiMPL with the lumped-mass metric,
# a stopping criterion, checkpoints and VTU output. The setup of top88 (Andreassen et al., 2011): unit square
# elements, plane stress, unit point load at the top-left node.
#
# Run: julia --project=examples -t auto examples/mbb.jl
using Gridap, HyperFEM, HyperShape

function mbb(; nx = 60, ny = 20, maxiter = 300, out = "results/mbb", verbose = 2)
  # Mesh, measures and spaces (Gridap and HyperFEM)
  model  = CartesianDiscreteModel((0, nx, 0, ny), (nx, ny))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "sym", [1, 3, 7])           # x = 0: u₁ = 0
  add_tag_from_tags!(labels, "roller", [2])              # bottom-right corner: u₂ = 0
  add_tag_from_tags!(labels, "top", [3, 6])
  Ω, Γ   = Triangulation(model), BoundaryTriangulation(model; tags = "top")
  dΩ, dΓ = Measure(Ω, 2), Measure(Γ, 2)
  bc = BoundaryConditions(BoundaryCondition("sym", VectorValue(0.0, 0.0); mask = (true, false)),
                          BoundaryCondition("roller", VectorValue(0.0, 0.0); mask = (false, true)))
  V  = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1), bc; conformity = :H1)
  U  = TrialFESpace(V, bc, 1.0)
  Vρ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)     # P0 design field

  # Design variables: z → x (identity) → θ = projection ∘ filter
  proj = TanhProjection(β = 1.0, η = 0.5)
  z, x, θ = single_field_design(:ρ, Vρ; bounds = (0.0, 1.0), init = 0.5,
                                map = proj ∘ HelmholtzFilter(dΩ; rmin = 0.04nx, ls = LUSolver()))

  # Physics: the weak forms of the user
  Emin, ν = 1e-9, 0.3
  λ, μ    = ν / (1 - ν^2), 1 / (2(1 + ν))                # plane stress
  E(ρ)    = Emin + ρ^3 * (1 - Emin)
  dE(ρ)   = 3ρ^2 * (1 - Emin)
  σ(ε)    = λ * tr(ε) * one(ε) + 2μ * ε
  t(x)    = VectorValue(0.0, x[1] < 1 ? -(4 - 6x[1]) : 0.0)   # unit point load at the top-left node
  a(u, v) = ∫((E ∘ θ.ρ) * (ε(v) ⊙ (σ ∘ ε(u))))dΩ
  l(v)    = ∫(t ⋅ v)dΓ
  S = StateOperator(StaticLinearModel(l, a, U, V, bc; ls = LUSolver(), backend = :compiled), θ)

  # Objective (compliance, self-adjoint) and constraint
  C   = Functional((u, θ) -> ∫(t ⋅ u)dΓ, S; adjoint = :self, scale = :initial,
                   derivative = (u, p, θ) -> (δ -> ∫(-(dE ∘ θ.ρ) * (ε(p) ⊙ (σ ∘ ε(u))) * δ)dΩ))
  Vol = Volume(θ, dΩ; max_fraction = 0.5)
  verbose > 0 && display(check_consistency((compliance = C, volume = Vol)))   # Taylor tests of the derivatives

  # Output every 10 iterations, checkpoints, and the optimisation
  post = OptPostProcessor(every = 10, filepath = joinpath(out, "vtk")) do post, ctx
    write_step(post, Ω, "rho" => θ.ρ, "x" => x.ρ, "u" => get_state(S))
  end
  problem = Optimization(objective = C, constraints = (volume = Vol,))
  solve!(problem, DensityMethod(SiMPL(); metric = LumpedMass(dΩ)); maxiter, verbose,
         stop         = StopCriteria(stationarity_rtol = 1e-3),
         continuation = Continuation(β = proj => [1, 2, 4, 8, 16]; every = 40),
         checkpoint   = Checkpoint(filepath = joinpath(out, "checkpoint"), every = 20),
         monitors     = (grey = Grey(θ, dΩ),), post)
end

if abspath(PROGRAM_FILE) == @__FILE__
  result = mbb()
  println(result)
end
