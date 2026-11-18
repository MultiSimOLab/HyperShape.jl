# Roadmap

| Phase | Content |
|---|---|
| 0. Foundations | Variables, operators, functionals, common loop, Helmholtz filter, optimality criteria and SiMPL, consistency checks. Validation on linear benchmarks (done) |
| 1. General constraints and reaction–diffusion | Reverse sweep with one adjoint solve per state, several constraints and an augmented Lagrangian, frozen regions, interface maps, reaction–diffusion step, discretisation contract |
| 2a. Nonlinear and coupled | Constitutive wrappers, robustness of the forward solve, chains of states and staggered adjoints, composite functionals, global and vector design variables, GCMMA, 3D |
| 2b. Formulations | Robust projection, stress constraints, design-dependent loads, manufacturing constraints as maps |
| 2c. Path dependence and time | Backward adjoint over load or time steps with internal variables (viscoelasticity, transient actuation) |
| 3. MPI on fixed meshes | Distributed design vectors, filters, optimisers and reaction–diffusion |
| 4. Advanced reaction–diffusion and adaptive meshes | Double well, local volume (maximum member size), adaptive and two-mesh discretisations |
| 5. Real geometries | STL/CAD to signed distance, non-design regions, Cartesian background meshes, multigrid |
| 6. Instabilities and eigenvalues | Buckling, eigenfrequencies, limit points |
| 7. Octree meshes and multigrid | Adaptive octree meshes with hanging nodes, geometric multigrid |
| 8. GPU | Matrix-free operators and multigrid on GPU |

Level-set evolution by PDE, cut-cell integration and body-fitted remeshing are research lines.
