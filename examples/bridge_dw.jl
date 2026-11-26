# Bridge 2 × 1 (Takezawa, Nishiwaki and Kitamura 2010): compliance minimisation with the reaction–diffusion method and
# the double-well reaction. The design field φ ∈ [0, 1] is a density read through Identity() with k(φ) = max(φ³, 1e-5)
# in the weak form; the double well keeps it close to 0/1 and only moves its interface, so the start has holes
# (hole_seed, the default :hex pattern). Roller at (0, 0), pin at (2, 0), a unit load downwards spread over the two
# cells of the bottom edge around x = 1; volume fraction 0.3, reached from the seed (1 % of the domain per iteration).
# Plane stress, E = 1, ν = 0.3. The diffusion κ = τ ℓ² = 1e-5 (that of Takezawa et al.) and the time per iteration
# T = 25 of the well are physical: the same on any mesh (the substeps per iteration grow like 1/h²).
# The published bridge (unit point load, their seed, 100 iterations: J = 23.3) is validation/bridge_takezawa.jl.
#
# Run: julia --project=examples -t auto examples/bridge_dw.jl
using Gridap, HyperFEM, HyperShape

function run_bridge_dw(; nx = 200, ny = 100, maxiter = 300, out = nothing, verbose = 2)
  # Mesh, measures and spaces (Gridap and HyperFEM)
  h = 2 / nx
  model  = CartesianDiscreteModel((0, 2, 0, 1), (nx, ny))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "roller", [1])                                   # the corner (0, 0)
  add_tag_from_tags!(labels, "pin", [2])                                      # the corner (2, 0)
  add_tag_from_vertex_filter!(labels, "load", model, x -> x[2] ≈ 0 && abs(x[1] - 1) <= h + 1e-9)
  Ω  = Triangulation(model)
  dΩ = Measure(Ω, 4)
  dΓ = Measure(BoundaryTriangulation(model; tags = "load"), 2)
  bc = BoundaryConditions(BoundaryCondition("roller", VectorValue(0.0, 0.0); mask = (false, true)),
                          BoundaryCondition("pin", VectorValue(0.0, 0.0)))
  V  = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1), bc; conformity = :H1)
  U  = TrialFESpace(V, bc, 1.0)
  Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)

  # Design: a density with holes (hex pattern), kept away from the supports and the load
  tags = ("roller", "pin", "load")
  φ0 = hole_seed(Vφ; bounds = (0.0, 1.0), radius = 0.1, spacing = 0.4, exclude = tags, connect = tags)
  z  = OptVariables(φ = FieldVariable(Vφ; bounds = (0.0, 1.0), init = φ0))
  x  = DesignFields(z; φ = Identity())
  θ  = PhysicalFields(x; χ = :φ => Identity())

  # Physics: k(φ) = max(φ³, 1e-5), plane stress; a unit load on the two cells around x = 1
  ν = 0.3
  λ, μ = ν / (1 - ν^2), 1 / (2(1 + ν))
  σ(ε) = λ * tr(ε) * one(ε) + 2μ * ε
  k(φ)  = max(φ^3, 1e-5)
  dk(φ) = φ^3 > 1e-5 ? 3φ^2 : 0.0
  t = VectorValue(0.0, -1 / (2h))
  a(u, v) = ∫((k ∘ θ.χ) * (ε(v) ⊙ (σ ∘ ε(u))))dΩ
  l(v) = ∫(t ⋅ v)dΓ
  S = StateOperator(StaticLinearModel(l, a, U, V, bc; ls = LUSolver(), backend = :compiled), θ;
        adjoint = AdjointProblem(sensitivity = (u, p) -> (δ -> ∫((dk ∘ θ.χ) * (ε(p) ⊙ (σ ∘ ε(u))) * δ)dΩ)))

  # Objective (compliance, self-adjoint) and volume
  C   = Functional((u, θ) -> ∫(t ⋅ u)dΓ, S; dJdu = SelfAdjoint(), scale = 1.0)
  Vol = Volume(θ, dΩ; max_fraction = 0.3)

  post = out === nothing ? nothing : OptPostProcessor(every = 20, filepath = joinpath(out, "vtk")) do post, ctx
    write_step(post, Ω, "phi" => x.φ, "u" => get_state(S))
  end
  problem = Optimization(z; objective = C, constraints = (volume = Vol,))
  κ = 1e-5
  method = ReactionDiffusion(; τ = κ / 2, reaction = DoubleWell())     # τ = κ/ℓ², ℓ² = |D| = 2; T = 25 per iteration
  solve!(problem, method; maxiter, verbose, post)          # default stop (after the settling) or maxiter
end

if abspath(PROGRAM_FILE) == @__FILE__
  result = run_bridge_dw(; out = "results/bridge_dw")
  println(result)
end
