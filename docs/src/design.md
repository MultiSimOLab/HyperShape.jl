# Design

!!! note
    This page describes the planned interface. Names and signatures may still change.

```
OptVariables z ─► DesignFields x ─► PhysicalFields θ ─► weak forms ─► StateOperator (u, p)
      ▲                                                                     │
      └──────── method (step) ◄── dJ/dz ◄── dJ/dθ ◄── Functional (J, DJ) ◄──┘
```

## Variables

Three linked levels, built in order; each one can only be created from the previous one.

| Level | Object | Meaning |
|---|---|---|
| `z` | `OptVariables` | What the optimiser sees: a block vector, scaled and bounded |
| `x` | `DesignFields` | Design fields on the design mesh: nodal (P1) or cell-wise (P0) |
| `θ` | `PhysicalFields` | Physical fields read by the weak forms at the quadrature points |

- A block of `z` is a field (density, level set, magnetisation direction, material fraction) or a global parameter
  (applied voltage, applied field, fibre angle).
- The map `z → x` holds scaling, frozen regions and symmetries. The map `x → θ` holds the filter, the projection and,
  optionally, the material interpolation.
- One design field may feed several physical fields, and one physical field may depend on several design fields.
- Bounds, initial values and maps are mandatory; write `Identity()` when there is no transformation.
- Only HyperShape writes `x` and `θ`; the user reads them as Gridap fields.

## Constitutive wrappers

The design enters the physics through constitutive models that wrap HyperFEM models by aggregation and return, besides
the energy and its derivatives, the analytical derivatives with respect to the physical fields (energy interpolation,
SIMP/RAMP per phase, interpolation of the solid contribution only in electro- and magneto-mechanics).

## Operators

Two kinds of operator wrap a HyperFEM computational model built by the user.

| | `StateOperator` (forward, adjoint) | `LinearOperator` (filter, reaction–diffusion step, extension) |
|---|---|---|
| Model | full weak forms | bilinear form only (the matrix) |
| Right-hand side | from the user's forms | vector assembled by HyperShape |
| `depends` | invalidates the result | invalidates the matrix |
| User function | `solve = (m, ctx) -> …` | `solve = (m, b, ctx, stale) -> …` |

- Dependencies are declared explicitly with `depends`; an operator solves again only when one of them has changed.
- `ctx` is a read-only context updated by HyperShape: design iteration, attempt, continuation values and phase.
- The adjoint is another state operator; its forms are written by the user, typically the tangent of the forward
  problem with trial and test functions swapped.

## Functionals

One `Functional` per objective or constraint:

```julia
J = Functional(Jform, DJform, S, A, θ; scale)
#   Jform(u, θ)       scalar value
#   DJform(u, p, θ)   total derivative with respect to θ, as a linear form
```

The value applies the forward operator `S`; the derivative applies the adjoint operator `A` and assembles `DJform`.
HyperShape then applies the transposed maps down to `z`.

## Methods

A method is an operator `z ↦ z⁺` over the functionals. All methods share one loop (update the variables, evaluate,
differentiate, step, accept or reject, continuation and stopping); only the step is specific.

| Method | Design field | Step |
|---|---|---|
| `SIMP(optimizer)` | density in [0, 1] | any `Optimizer` |
| `ReactionDiffusion` | level set in [−1, 1] | one linear solve, then clipping |
| `LevelSet` | signed distance | velocity extension, advection, reinitialisation |

If the forward problem does not converge for a design, the step is rejected and shortened; after a maximum number of
attempts the optimisation stops with a checkpoint.

## Optimisers

`HyperShape.Optimizers` is independent of finite elements: an optimiser sees vectors, bounds and an oracle.

```julia
abstract type Optimizer end
init(opt, z, oracle)               -> state
step!(opt, state, z, oracle, ctx)  -> status
reset!(opt, state)
shrink!(opt, state, factor)
```

## Post-processing

`OptPostProcessor(problem, driver; every, filepath)` calls the user-written `driver(post, ctx)` every `every` design
iterations, following the pattern of HyperFEM's `PostProcessor`. The user decides which fields are written and on
which triangulation, including domains larger than the design region.
