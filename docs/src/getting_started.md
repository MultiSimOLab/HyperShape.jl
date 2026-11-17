# Getting started

## Installation

HyperShape is not registered yet. Add the `main` branches of HyperFEM and HyperFEMKernels first (HyperShape follows
them during development) and then HyperShape:

```julia
pkg> add https://github.com/MultiSimOLab/HyperFEM.jl#main https://github.com/MultiSimOLab/HyperFEMKernels.jl#main
pkg> add https://github.com/MultiSimOLab/HyperShape.jl
```

Start Julia with several threads (`julia -t auto`): HyperShape assembles its vectors with the compiled, threaded
kernels of HyperFEMKernels.

## A complete script

The half MBB beam of top88 (Andreassen et al., 2011): minimise the compliance with a volume constraint, using a density
method with a SIMP interpolation, a Helmholtz filter and a tanh projection whose sharpness grows in stages. The script
is `examples/mbb.jl` in the repository; the test suite runs it at a small size.

```julia
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
```

## What each part does

- **Mesh, spaces and boundary conditions** are plain Gridap and HyperFEM: HyperShape adds nothing there.
- **Design variables.** `single_field_design` builds the three levels: the optimisation variables `z` (one value per
  cell here, bounds `[0, 1]`), the design field `x` and the physical field `θ.ρ = proj(filter(x))` read by the weak
  forms. `HelmholtzFilter(dΩ; rmin, ls)` is the PDE filter with radius `rmin`; `ls` is its linear solver (yours to
  choose). Maps compose with `∘` in the mathematical order. Other maps: `CellMean` (one value per cell, as top88),
  `PointwiseMap(f, df)` (your own), or a filter on a `LinearOperator` of your own model.
- **Physics.** The weak forms are yours: here linear elasticity with `E(ρ) = Emin + ρ³(1 − Emin)`. The state is a
  `StateOperator` over a HyperFEM model that depends on `θ`: it is solved again only when `θ` changes.
- **Functionals.** `Functional(J, S; derivative, adjoint, scale)`: the value and its derivative with respect to the
  physical fields, written by you (the compliance is self-adjoint: `adjoint = :self`). `Volume` is built in.
  `check_consistency` runs Taylor tests of every derivative and checks the declared dependencies: run it once on a
  coarse mesh whenever you write a new derivative.
- **Solve.** `Optimization(objective, constraints)` is the problem; `DensityMethod(SiMPL(); metric)` the method (or
  `OC()`). `solve!` runs the common loop: `stop` criteria (here the stationarity measure of the SiMPL paper,
  relative to the first iteration), `continuation` of parameters in stages (β of the projection; any `Parameter` of
  yours, such as the SIMP penalty), `checkpoint` (resume with `solve!(…; resume = folder)`), `monitors` (the measure of
  grey) and `post` (your output, here VTU files every 10 iterations, without blocking the loop).
- **Result.** `solve!` returns an `OptResult`: `status`, `reason`, `iterations`, `objective`, the variables at the
  final design (`get_values(result.z)`) and the `history` (`history.objective`, `history.constraints.volume`,
  `history.monitors.grey`, `history.continuation.β`, …).

The screen shows a banner with the setup (each default marked), one row per iteration and a summary of the solves.
