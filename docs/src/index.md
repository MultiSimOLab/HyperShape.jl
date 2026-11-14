```@meta
CurrentModule = HyperShape
```

# HyperShape.jl

!!! warning "Work in progress"
    The package is at the design stage. These pages describe the planned architecture; only a small legacy
    prototype is implemented today.

[HyperShape](https://github.com/MultiSimOLab/HyperShape.jl) is a library for shape and topology optimisation of
nonlinear multiphysics problems: finite-strain hyperelasticity, electro-, magneto- and thermo-mechanics. It is built
on [HyperFEM](https://github.com/MultiSimOLab/HyperFEM.jl), within the [Gridap](https://github.com/gridap/Gridap.jl)
ecosystem.

## Principles

- **HyperFEM is used as a black box.** The forward and adjoint problems are HyperFEM computational models that the
  user builds with their own weak forms.
- **The user configures every solver.** HyperShape chooses none: not for the state, the adjoint, the filter or the
  internal solves of the methods.
- **Analytical derivatives.** No automatic differentiation; every sensitivity is verified with a Taylor test.
- **One abstraction for several methods.** Density (SIMP), reaction–diffusion and level set share variables, states,
  functionals and the optimisation loop.

## Contents

- [Design](@ref): the planned architecture.
- [Roadmap](@ref): development phases.
- [API reference](@ref): docstrings of what is implemented.
