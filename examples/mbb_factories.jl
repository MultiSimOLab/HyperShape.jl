# The half MBB beam of mbb.jl written with the two factories of `Optimization(model; design_setup, physics_setup)`:
# the same physics, the same run, but the problem can be built again by solve! (discretization). Today this serves to
# check the factories (`HyperShape.Methods.RebuildCheck` gives the run of `FixedMesh()` bit for bit); with adaptive
# meshes (later) the script stays the same.
#
# Run: julia --project=examples -t auto examples/mbb_factories.jl
using Gridap, HyperFEM, HyperShape

function run_mbb_factories(; nx = 60, ny = 20, maxiter = 300, out = "results/mbb_factories", verbose = 2,
                           discretization = FixedMesh())
  # Mesh (Gridap): the tags are put on the model once, outside the factories
  model  = CartesianDiscreteModel((0, nx, 0, ny), (nx, ny))
  labels = get_face_labeling(model)
  add_tag_from_vertex_filter!(labels, "sym", model, x -> x[1] ≈ 0)                    # x = 0: u₁ = 0
  add_tag_from_vertex_filter!(labels, "roller", model, x -> x[1] ≈ nx && x[2] ≈ 0)    # bottom-right corner: u₂ = 0
  add_tag_from_vertex_filter!(labels, "load", model, x -> x[2] ≈ ny && x[1] <= 1)     # first edge of the top

  # Continued and independent of the mesh: created once, outside the factories, used inside
  proj = TanhProjection(β = 1.0, η = 0.5)

  # Design: space and variables, z → x (identity)
  function design_setup(model)
    Vρ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)    # P0 design field
    z  = OptVariables(ρ = FieldVariable(Vρ; bounds = (0.0, 1.0), init = 0.5))
    x  = DesignFields(z; ρ = Identity())
    (; z, x)
  end

  # Physics: x → θ = projection ∘ filter, the state, the functionals, the monitors and the output (the code of mbb.jl)
  function physics_setup(model, x)
    Ω, Γ   = Triangulation(model), BoundaryTriangulation(model; tags = "load")
    dΩ, dΓ = Measure(Ω, 2), Measure(Γ, 2)
    bc = BoundaryConditions(BoundaryCondition("sym", VectorValue(0.0, 0.0); mask = (true, false)),
                            BoundaryCondition("roller", VectorValue(0.0, 0.0); mask = (false, true)))
    V  = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1), bc; conformity = :H1)
    U  = TrialFESpace(V, bc, 1.0)
    θ  = PhysicalFields(x; ρ = proj ∘ HelmholtzFilter(dΩ; rmin = 0.04nx, ls = LUSolver()))

    # SIMP (p = 3) by the energy interpolation of HyperFEM, plane stress, ν = 0.3
    ν, Emin = 0.3, 1e-9
    λ, μ    = ν / (1 - ν^2), 1 / (2(1 + ν))
    solid   = LinearElasticity2D(λ = λ, μ = μ)
    void    = LinearElasticity2D(λ = Emin * λ, μ = Emin * μ)
    _, _, ∂²Ψ, _, _, ∂²Ψ_∂ρ = EnergyInterpolationScheme(solid, void; p = 3.0)()
    F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
    t(x)    = VectorValue(0.0, -(4 - 6x[1]))      # unit point load at the top-left node
    a(u, v) = ∫(∇(v)' ⊙ ((∂²Ψ ∘ (θ.ρ, F ∘ ∇(u)')) ⊙ ∇(u)'))dΩ
    l(v)    = ∫(t ⋅ v)dΓ
    S = StateOperator(StaticLinearModel(l, a, U, V, bc; ls = LUSolver(), backend = :compiled), θ;
          adjoint = AdjointProblem(sensitivity = (u, p) -> (δ -> ∫((∇(p)' ⊙ ((∂²Ψ_∂ρ ∘ (θ.ρ, F ∘ ∇(u)')) ⊙ ∇(u)')) * δ)dΩ)))

    C   = Functional((u, θ) -> ∫(t ⋅ u)dΓ, S; dJdu = SelfAdjoint(), scale = :initial)
    Vol = Volume(θ, dΩ; max_fraction = 0.5)
    (; objective = C, constraints = (volume = Vol,), θ, S,                  # θ and S: result.problem.physics.θ
       monitors = (grey = Grey(θ, dΩ),),
       output   = (post, ctx) -> write_step(post, Ω, "rho" => θ.ρ, "x" => x.ρ, "u" => get_state(S)))
  end

  problem = Optimization(model; design_setup, physics_setup)
  verbose > 0 && display(check_consistency(problem))                # Taylor tests of the derivatives

  post = OptPostProcessor(every = 10, filepath = joinpath(out, "vtk"))   # no driver: the output of physics_setup
  solve!(problem, DensityMethod(SiMPL(); metric = LumpedMass()); maxiter, verbose, post, discretization,
         stop         = StopCriteria(stationarity_rtol = 1e-3),
         continuation = Continuation(β = proj => [1, 2, 4, 8, 16]; every = 40),
         checkpoint   = Checkpoint(filepath = joinpath(out, "checkpoint"), every = 20))
         # discretization = HyperShape.Methods.RebuildCheck(every = 25) checks the factories
end

if abspath(PROGRAM_FILE) == @__FILE__
  result = run_mbb_factories()
  println(result)
end
