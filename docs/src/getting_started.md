# Getting started

## Installation

HyperShape is not registered yet. Add the `main` branches of HyperFEM and HyperFEMKernels first (HyperShape follows
them during development) and then HyperShape:

```julia
pkg> add Gridap@0.20.8
pkg> add https://github.com/MultiSimOLab/HyperFEM.jl#main https://github.com/MultiSimOLab/HyperFEMKernels.jl#main
pkg> add https://github.com/MultiSimOLab/HyperShape.jl
```

It requires Julia 1.12 or later and Gridap 0.20.8. Start Julia with several threads (`julia -t auto`): HyperShape
assembles its vectors with the compiled, threaded kernels of HyperFEMKernels, and writes its output in another thread.

## A complete script

The half MBB beam of top88 (Andreassen et al., 2011): minimise the compliance with a volume constraint, using a density
method with a SIMP interpolation, a Helmholtz filter and a tanh projection whose sharpness grows in stages. The script
is `examples/mbb.jl` in the repository (run it with `julia --project=examples -t auto examples/mbb.jl` after
`julia --project=examples -e 'using Pkg; Pkg.instantiate()'`); the test suite runs it at a small size. On a laptop the
60×20 beam converges in about 200 iterations and a few seconds.

```julia
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
  # the state, and how to differentiate through it: pᵀ ∂R/∂ρ [δ] at the state u and the adjoint p
  S = StateOperator(StaticLinearModel(l, a, U, V, bc; ls = LUSolver(), backend = :compiled), θ;
        adjoint = AdjointProblem(sensitivity = (u, p) -> (δ -> ∫((∇(p)' ⊙ ((∂²Ψ_∂ρ ∘ (θ.ρ, F ∘ ∇(u)')) ⊙ ∇(u)')) * δ)dΩ)))

  # Objective (compliance, self-adjoint: p = u, no adjoint solve) and constraint
  C   = Functional((u, θ) -> ∫(t ⋅ u)dΓ, S; dJdu = SelfAdjoint(), scale = :initial)
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

result = run_mbb()
```

## What each part does

- **Mesh, spaces and boundary conditions** are plain Gridap and HyperFEM: HyperShape adds nothing there.
- **Design variables.** `single_field_design` builds the three levels: the optimisation variables `z` (one value per
  cell here, bounds `[0, 1]`), the design field `x` and the physical field `θ.ρ = proj(filter(x))` read by the weak
  forms. `HelmholtzFilter(dΩ; rmin, ls)` is the PDE filter with radius `rmin`; `ls` is its linear solver (yours to
  choose). Maps compose with `∘` in the mathematical order. Other maps: `CellMean` (one value per cell, as top88),
  `PointwiseMap(f, df)` (your own), or a filter on a `LinearOperator` of your own model.
- **Mesh tags** with HyperFEM's `add_tag_from_vertex_filter!` (an entity is tagged when all its vertices pass the
  filter); `CartesianTags` has the tags of the faces, edges and corners of a Cartesian box.
- **Physics.** The weak forms are yours, built here from constitutive models of HyperFEM: the
  `EnergyInterpolationScheme` between a solid and a weak void (`LinearElasticity2D` with stiffness `Emin`), whose
  energy is `ρ³ Ψs + (1 − ρ³) Ψv` (SIMP with p = 3). It returns the derivatives with respect to `F = I + ∇u` and to
  `ρ`: the second derivative in `F` gives the bilinear form, its derivative in `ρ` the sensitivity of the state. The state is a
  `StateOperator` over a HyperFEM model that depends on `θ`: it is solved again only when `θ` changes. Its
  `AdjointProblem` says how to differentiate through it: the `sensitivity` `pᵀ ∂R/∂θ [δ]` at the state `u` and the
  adjoint `p`, written once per state and without sign, and who solves the transposed system. Here nothing is solved
  (the compliance is self-adjoint); for other functionals give `SymmetricState()` (reuses the factorisation of a linear
  and symmetric state) or a model of yours with the transposed tangent.
- **Functionals.** `Functional(J, S; dJdu, dJdθ, scale)`: the value and its partial derivatives with respect to the
  state (`dJdu = u -> (w -> ∫(…)dΩ)`, or `SelfAdjoint()` for the compliance, so `p = u` and no adjoint is solved) and
  to the physical fields (`dJdθ`, only if `J` reads them). HyperShape puts the signs. `Volume` is built in.
  `check_consistency` runs Taylor tests of every derivative and checks the declared dependencies: run it once on a
  coarse mesh whenever you write a new derivative.
- **Solve.** `Optimization(objective = C, constraints = (volume = Vol,))` is the problem; `DensityMethod(SiMPL(); metric)` the method (or
  `OC()`). `solve!` runs the common loop: `stop` criteria (here the stationarity measure of the SiMPL paper,
  relative to the first iteration), `continuation` of parameters in stages (β of the projection; any `Parameter` of
  yours, such as the SIMP penalty, read in the weak forms and declared as a dependency of the state:
  `StateOperator(model, (θ, p))`), `checkpoint` (resume with `solve!(…; resume = folder)`), `monitors` (the measure of
  grey) and `post` (your output, here VTU files every 10 iterations, written in another thread when Julia has several).
  With `OC()` instead of `SiMPL()` the same script needs more iterations to meet this stationarity tolerance.
- **Result.** `solve!` returns an `OptResult`: `status`, `reason`, `iterations`, `objective` (scaled: divided by the
  first value, `scale = :initial`), the variables at the
  final design (`get_values(result.z)`) and the `history` (`history.objective`, `history.constraints.volume`,
  `history.monitors.grey`, `history.continuation.β`, …).

The screen shows a banner with the setup (each default marked), one row per iteration and a summary of the solves.
