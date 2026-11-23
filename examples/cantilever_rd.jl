# Cantilever 2 × 1: compliance minimisation with the reaction–diffusion method. The design field φ ∈ [−1, 1] is a
# level set (φ ≥ 0 solid) read through the ersatz material of Allaire, Jouve and Toader (Ersatz(): the fraction of each
# cell where φ ≥ 0); the material is E(χ) = Emin + χ (1 − Emin). Clamped on the left side, unit traction downwards on
# the middle fifth of the right side, with the cells under the load frozen as solid; volume fraction 0.5, reached
# from a solid start (1 % of the domain per iteration). Plane stress, E = 1, ν = 0.3.
#
# Run: julia --project=examples -t auto examples/cantilever_rd.jl
using Gridap, HyperFEM, HyperShape

function run_cantilever_rd(; nx = 120, ny = 60, maxiter = 400, interface = Ersatz(), τ = 2e-4, out = nothing, verbose = 2)
  # Mesh, measures and spaces (Gridap and HyperFEM)
  model  = CartesianDiscreteModel((0, 2, 0, 1), (nx, ny))
  labels = get_face_labeling(model)
  add_tag_from_vertex_filter!(labels, "clamp", model, x -> x[1] ≈ 0)
  add_tag_from_vertex_filter!(labels, "load", model, x -> x[1] ≈ 2 && abs(x[2] - 0.5) <= 0.1 + 1e-9)
  tag_adjacent_cells!(labels, "load_zone", model, "load")                     # the cells under the load
  Ω, Γ = Triangulation(model), BoundaryTriangulation(model; tags = "load")
  dΓ = Measure(Γ, 2)
  bc = BoundaryConditions(BoundaryCondition("clamp", VectorValue(0.0, 0.0)))
  V  = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1), bc; conformity = :H1)
  U  = TrialFESpace(V, bc, 1.0)
  Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1, dirichlet_tags = "load_zone")

  # Design: level set φ ∈ [−1, 1], solid start, frozen at 1 under the load; χ = material through the interface map
  z  = OptVariables(φ = FieldVariable(Vφ; bounds = (-1.0, 1.0), init = 1.0, frozen = 1.0))
  x  = DesignFields(z; φ = Identity())
  θ  = PhysicalFields(x; χ = :φ => interface)
  dΩ = design_measures(θ; degree = 4)

  # Physics: ersatz material between the solid and a weak void, plane stress
  ν, Emin = 0.3, 1e-3
  λ, μ = ν / (1 - ν^2), 1 / (2(1 + ν))
  σ(ε) = λ * tr(ε) * one(ε) + 2μ * ε
  E(χ) = Emin + χ * (1 - Emin)
  t = VectorValue(0.0, -1.0)
  a(u, v) = ∫((E ∘ θ.χ) * (ε(v) ⊙ (σ ∘ ε(u))))dΩ.domain
  l(v) = ∫(t ⋅ v)dΓ
  # pᵀ ∂R/∂χ [δ] at the state u and the adjoint p (the exact derivative: the method weights it with the material)
  S = StateOperator(StaticLinearModel(l, a, U, V, bc; ls = LUSolver(), backend = :compiled), θ;
        adjoint = AdjointProblem(sensitivity = (u, p) -> (δ -> ∫((1 - Emin) * (ε(p) ⊙ (σ ∘ ε(u))) * δ)dΩ.domain)))

  # Objective (compliance, self-adjoint) and volume
  C   = Functional((u, θ) -> ∫(t ⋅ u)dΓ, S; dJdu = SelfAdjoint(), scale = :initial)
  Vol = Volume(θ, dΩ; max_fraction = 0.5)

  post = out === nothing ? nothing : OptPostProcessor(every = 20, filepath = joinpath(out, "vtk")) do post, ctx
    write_step(post, Ω, "chi" => θ.χ, "phi" => x.φ, "u" => get_state(S))
  end
  problem = Optimization(z; objective = C, constraints = (volume = Vol,))
  method  = ReactionDiffusion(; τ, sensitivity = :physical)
  solve!(problem, method; maxiter, verbose, post)       # stop: ‖Δφ‖∞ < 1e-2 with the volume met (default)
end

if abspath(PROGRAM_FILE) == @__FILE__
  result = run_cantilever_rd(; out = "results/cantilever_rd")
  println(result)
end
