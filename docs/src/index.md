```@meta
CurrentModule = HyperShape
```

# HyperShape.jl

[HyperShape](https://github.com/MultiSimOLab/HyperShape.jl) is a library for shape and topology optimisation of
nonlinear multiphysics problems: finite-strain hyperelasticity, electro-, magneto- and thermo-mechanics. It is built
on [HyperFEM](https://github.com/MultiSimOLab/HyperFEM.jl), within the [Gridap](https://github.com/gridap/Gridap.jl)
ecosystem.

!!! note "Status"
    Phase 0 is implemented and validated: density methods for linear problems (Helmholtz filter, tanh projection,
    optimality criteria and SiMPL, consistency checks, continuation, checkpoints, fast output), reproducing top88 and
    the SiMPL paper (see [Validation](@ref)). The reaction–diffusion method (level set through an interface map, or
    density) is implemented. Nonlinear and multiphysics problems and level-set methods come in the next phases
    ([Roadmap](@ref)). The package is not registered yet.

## Principles

- **HyperFEM is used as a black box.** The forward and adjoint problems are HyperFEM computational models that the
  user builds with their own weak forms.
- **The user configures every solver.** HyperShape chooses none: not for the state, the adjoint, the filter or the
  internal solves of the methods. Every default of the loop can be replaced from the script.
- **Analytical derivatives.** No automatic differentiation; every sensitivity is verified with a Taylor test
  (`check_consistency`).
- **One abstraction for several methods.** Density, reaction–diffusion and level set are meant to share variables,
  states, functionals and the optimisation loop (density and reaction–diffusion methods are implemented).

## Contents

- [Getting started](@ref): a complete script, explained.
- [Design](@ref): the architecture and the defaults.
- [Validation](@ref): benchmarks against published results.
- [Roadmap](@ref): development phases.
- [API reference](@ref): docstrings.
