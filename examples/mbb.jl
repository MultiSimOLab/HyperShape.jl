# Half MBB beam: compliance minimisation with a density method (SIMP interpolation, p = 3), a Helmholtz filter, a
# tanh projection with continuation, a volume constraint on the physical density, SiMPL with the lumped-mass metric,
# a stopping criterion, checkpoints and VTU output. The setup of top88 (Andreassen et al., 2011): unit square
# elements, plane stress, unit point load at the top-left node.
#
# Run: julia --project=examples -t auto examples/mbb.jl
using Gridap, HyperFEM, HyperShape

function run_mbb(; nx = 60, ny = 20, maxiter = 300, out = "results/mbb", verbose = 2)
  # Mesh, measures and spaces (Gridap and HyperFEM)
  model  = CartesianDiscreteModel((0, nx, 0, ny), (nx, ny))
  labels = get_face_labeling(model)
  add_tag_from_vertex_filter!(labels, "sym", model, x -> x[1] ≈ 0)                    # x = 0: u₁ = 0
  add_tag_from_vertex_filter!(labels, "roller", model, x -> x[1] ≈ nx && x[2] ≈ 0)    # bottom-right corner: u₂ = 0
  add_tag_from_vertex_filter!(labels, "load", model, x -> x[2] ≈ ny && x[1] <= 1)     # first edge of the top
  Ω, Γ   = Triangulation(model), BoundaryTriangulation(model; tags = "load")
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

  # Physics: energy interpolation of HyperFEM between the solid and a weak void (SIMP, p = 3), plane stress, ν = 0.3
  ν, Emin = 0.3, 1e-9
  λ, μ    = ν / (1 - ν^2), 1 / (2(1 + ν))
  solid   = LinearElasticity2D(λ = λ, μ = μ)
  void    = LinearElasticity2D(λ = Emin * λ, μ = Emin * μ)
  _, _, ∂²Ψ, _, _, ∂²Ψ_∂ρ = EnergyInterpolationScheme(solid, void; p = 3.0)()   # ρ³ Ψs + (1 − ρ³) Ψv and ∂/∂ρ
  F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))                            # F(∇u) = I + ∇u
  t(x)    = VectorValue(0.0, -(4 - 6x[1]))      # unit point load at the top-left node (dual to its shape function)
  a(u, v) = ∫(∇(v)' ⊙ ((∂²Ψ ∘ (θ.ρ, F ∘ ∇(u)')) ⊙ ∇(u)'))dΩ
  l(v)    = ∫(t ⋅ v)dΓ
  S = StateOperator(StaticLinearModel(l, a, U, V, bc; ls = LUSolver(), backend = :compiled), θ)

  # Objective (compliance, self-adjoint) and constraint
  C   = Functional((u, θ) -> ∫(t ⋅ u)dΓ, S; adjoint = :self, scale = :initial,
                   derivative = (u, p, θ) -> (δ -> ∫(-(∇(p)' ⊙ ((∂²Ψ_∂ρ ∘ (θ.ρ, F ∘ ∇(u)')) ⊙ ∇(u)')) * δ)dΩ))
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
  result = run_mbb()
  println(result)
end
