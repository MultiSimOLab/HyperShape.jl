# Roadmap

The detailed plan, with a checkbox per item, is kept in the
[development roadmap issue](https://github.com/MultiSimOLab/HyperShape.jl/issues/4).
Nonlinear and multiphysics problems are addressed only once the methods are stable on linear problems.

| Phase | Content | Status |
|---|---|---|
| 0. Foundations | Variables, operators, functionals, common loop, Helmholtz filter, optimality criteria and SiMPL, consistency checks, validation on linear benchmarks | Done |
| 1. General constraints and common loop | Reverse sweep with one adjoint solve per state, general constraints and augmented Lagrangian, frozen regions, interface maps (`Step()`, `Ersatz()`), discretisation contract (`FixedMesh()`) | Done |
| 1b. Consolidation on linear problems | Reaction–diffusion (level set, density, double well), initial hole seeding, benchmark suite with reference values (top88, levelset88, Kim et al. 2020), mesh independence, report of designs, closing 3D linear case | In progress |
| 1c. Constraints and formulations (linear) | GCMMA and field constraints, null-space optimiser, perimeter, stress constraints, minimum and maximum member size, self-weight | Planned |
| 1d. Adaptive meshing (ersatz material) | `TwoMeshes`, then `AdaptiveMesh`; transfer of the optimiser state at each remeshing | Planned |
| 2a. Nonlinear and coupled | Design-dependent constitutive models, robust forward solve, chains of states and staggered adjoints, global and vector design variables, 3D, multi-material with several level sets | Planned |
| 2b. Formulations on nonlinear problems | Phase 1c constraints on nonlinear and multiphysics problems, design-dependent loads, overhang constraints | Planned |
| 2c. Path dependence and time | Adjoint over load or time steps with internal variables (viscoelasticity, transient actuation) | Planned |
| 3. MPI on fixed meshes | Distributed design vectors, filters, optimisers and reaction–diffusion | Planned |
| 4. Body-fitted meshes and cut cells | `FittedMesh` and `CutCell()` with the reaction–diffusion level set | Planned |
| 5. Real geometries | STL/CAD to signed distance, non-design regions, Cartesian background meshes, STL output, body-fitted re-analysis | Planned |
| 6. Instabilities and eigenvalues | Buckling, eigenfrequencies, limit points, connectivity constraint | Planned |
| 7. Octree meshes and multigrid | Adaptive octree meshes with hanging nodes, geometric multigrid | Planned |
| 8. GPU | Matrix-free operators and multigrid on GPU | Planned |

To be scheduled: level set with topological derivative (Amstutz and Andrä 2006) after Phase 1b, and microstructure
design with linear homogenisation.

Research lines: classical (Hamilton–Jacobi) level set, methods as gradient flows, chaining of methods,
optimisation under uncertainty.
