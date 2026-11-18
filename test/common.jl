using Gridap
using Gridap.FESpaces
using HyperFEM
using HyperShape

# Heat-conduction-like problem: -∇⋅(k(θ.ρ) ∇u) = f in Ω, u = 0 on the left side, with k(ρ) = kmin + ρ³.
function state_problem(; n = 8, dim = 2, design = :P0, backend = :lazy, gD = 0.0, adjoint_space = :homogeneous,
                       vector_backend = :compiled, init = nothing)
  model = CartesianDiscreteModel(dim == 2 ? (0, 1, 0, 1) : (0, 1, 0, 1, 0, 1), ntuple(_ -> n, dim))
  labels = get_face_labeling(model)
  if dim == 2
    add_tag_from_tags!(labels, "left", [1, 3, 7])
    add_tag_from_tags!(labels, "right", [2, 4, 8])
  else
    add_tag_from_tags!(labels, "left", [25])     # face x = 0
    add_tag_from_tags!(labels, "right", [26])    # face x = 1
  end
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  Γ = BoundaryTriangulation(model; tags = "right")
  dΓ = Measure(Γ, 2)
  order = design === :P0 ? 0 : 1
  Vρ = TestFESpace(model, ReferenceFE(lagrangian, Float64, order); conformity = design === :P0 ? :L2 : :H1)
  Vf = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
  Uf = TrialFESpace(Vf)
  af(u, v) = ∫(0.01 * ∇(u) ⋅ ∇(v) + u * v)dΩ
  Sf = LinearOperator(StaticLinearModel(af, Uf, Vf, NothingBC(); ls = LUSolver()); depends = (),
                      solve = (m, b, ctx, stale) -> solve!(m, b; Assembly = stale))
  ρ0 = init === nothing ? [0.3 + 0.5 * abs(sin(2.3i)) for i in 1:num_free_dofs(Vρ)] : init
  H = TanhProjection(β = 3.0, η = 0.5)
  z, x, θ = single_field_design(:ρ, Vρ; bounds = (0.0, 1.0), init = ρ0, backend = vector_backend,
                                map = H ∘ HelmholtzFilter(Sf, dΩ; backend = vector_backend))

  kmin = 1e-2
  k(ρ) = kmin + ρ^3
  dk(ρ) = 3ρ^2
  f(x) = 1 + x[1]
  bc = BoundaryConditions(BoundaryCondition("left", gD))
  bc0 = BoundaryConditions(BoundaryCondition("left", 0.0))
  V = TestFESpace(Ω, ReferenceFE(lagrangian, Float64, 1), bc, conformity = :H1)
  U = TrialFESpace(V, bc, 1.0)
  U0 = TrialFESpace(V, bc0, 1.0)
  a(u, v) = ∫((k ∘ θ.ρ) * ∇(u) ⋅ ∇(v))dΩ
  l(v) = ∫(f * v)dΩ
  nforward, nadjoint, nassembly = Ref(0), Ref(0), Ref(0)
  ctxs = Any[]
  model_u = StaticLinearModel(l, a, U, V, bc; ls = LUSolver(), backend)
  uh = get_state(model_u)
  # pᵀ ∂R/∂θ [δ] of R(u)[v] = a(u, v) − l(v)
  sens = (u, q) -> (δ -> ∫((dk ∘ θ.ρ) * ∇(u) ⋅ ∇(q) * δ)dΩ)
  # adjoint: the matrix of a (symmetric) with test and trial swapped, on the homogeneous space
  Ua, bca = adjoint_space === :homogeneous ? (U0, bc0) : (U, bc)
  model_p = StaticLinearModel((p, w) -> a(w, p), Ua, V, bca; ls = LUSolver(), backend)
  adj = AdjointProblem(model_p; sensitivity = sens,
                       solve = (m, b, ctx, stale) -> (nadjoint[] += 1; nassembly[] += stale; push!(ctxs, (:A, ctx));
                                                      solve!(m, b; Assembly = stale)))
  S = StateOperator(model_u; depends = θ, adjoint = adj,
                    solve = (m, ctx) -> (nforward[] += 1; push!(ctxs, (:S, ctx)); solve!(m; Assembly = true)))
  # the two functionals of the tests: the compliance (self-adjoint) and the mean value on the right side
  compliance(; kw...) = Functional((u, θ) -> ∫(f * u)dΩ, S; dJdu = SelfAdjoint(), fields = θ, kw...)
  boundary(; kw...) = Functional((u, θ) -> ∫(u)dΓ, S; dJdu = u -> (w -> ∫(w)dΓ), fields = θ, kw...)
  (; z, x, θ, H, S, uh, k, dk, f, dΩ, dΓ, nforward, nadjoint, nassembly, Vρ, U, U0, V, bc, bc0, a, l, ctxs, Sf, sens,
     adj, model_p, compliance, boundary)
end

