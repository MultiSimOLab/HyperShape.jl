<p align="center"><img src="https://github.com/MultiSimOLab/HyperShape.jl/raw/main/docs/img/logo.png" width="30%"></p>

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://multisimolab.github.io/HyperShape.jl/dev/)
[![Build Status](https://github.com/MultiSimOLab/HyperShape.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MultiSimOLab/HyperShape.jl/actions/workflows/ci.yml?query=branch%3Amain)

## Shape and Topology Optimisation of Nonlinear Multiphysics

**HyperShape** is a library for **shape and topology optimisation of nonlinear multiphysics problems**
(finite-strain hyperelasticity, electro-, magneto- and thermo-mechanics). It is built on top of
[HyperFEM](https://github.com/MultiSimOLab/HyperFEM.jl), within the [Gridap](https://github.com/gridap/Gridap.jl)
ecosystem, and uses HyperFEM's computational models as black boxes for the forward and adjoint problems.

## Goals

- One abstraction for **density (SIMP)**, **reaction–diffusion** and **level-set** methods.
- **Weak forms, computational models and solvers are written and configured by the user**; HyperShape adds the
  design variables, the functionals, the sensitivities and the optimisation loop.
  Defaults (the solve of a linear model, stopping criteria, step rejection) can always be replaced from the script.
- **Analytical derivatives** throughout, verified with Taylor (finite-difference) tests.
- Structured meshes (real geometries imported from STL/CAD on octree background meshes) and unstructured meshes.
- Designed for large problems: in-place operations on generic vectors, user-supplied solvers, global operations
  as reductions only.

## Planned architecture

```
OptVariables z ─► DesignFields x ─► PhysicalFields θ ─► weak forms ─► StateOperator (u, p)
      ▲                                                                     │
      └──────── method (step) ◄── dJ/dz ◄── dJ/dθ ◄── Functional (J, DJ) ◄──┘
```

| Piece | Role |
|---|---|
| `OptVariables`, `DesignFields`, `PhysicalFields` | Three linked levels of variables: what the optimiser sees, the design fields (nodal or cell-wise), and the physical fields read by the weak forms |
| `StateOperator`, `AdjointOperator` | Forward and adjoint states over HyperFEM computational models (default `solve` for linear models) |
| `LinearOperator` | Auxiliary linear solves (filter, reaction–diffusion step, velocity extension) |
| `Functional` | Objective and constraints, with their derivative with respect to the physical fields |
| Methods | `SIMP`, `ReactionDiffusion`, `LevelSet`: a common loop where only the step differs |
| `Optimizers` | Optimisers independent of finite elements (optimality criteria, SiMPL, …) |
| `OptPostProcessor` | User-written output at each design iteration |

See the documentation for the design in more detail.

## Installation

HyperShape is not registered yet. To try the development version, add the `main` branches of HyperFEM and
HyperFEMKernels first (HyperShape follows them during development) and then HyperShape:

```julia
pkg> add https://github.com/MultiSimOLab/HyperFEM.jl#main https://github.com/MultiSimOLab/HyperFEMKernels.jl#main
pkg> add https://github.com/MultiSimOLab/HyperShape.jl
```

It requires Julia 1.12 or later and Gridap 0.20.8, and depends on HyperFEM and HyperFEMKernels. It cannot be
installed from the General registry until both are released with these changes.

Start Julia with several threads (`julia -t auto`): by default HyperShape assembles its vectors with the compiled,
threaded kernels of HyperFEMKernels.

## Related packages

- [HyperFEM.jl](https://github.com/MultiSimOLab/HyperFEM.jl): constitutive models, weak forms, computational models and solvers.
- [HyperFEMKernels.jl](https://github.com/MultiSimOLab/HyperFEMKernels.jl): assembly engine for HyperFEM.

## License

MIT. See [LICENSE](LICENSE).
