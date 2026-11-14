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
| `z` | `OptVariables` | What the optimiser sees: one vector made of named blocks, with bounds |
| `x` | `DesignFields` | Design fields on the design mesh: nodal (P1) or cell-wise (P0) |
| `θ` | `PhysicalFields` | Physical fields read by the weak forms at the quadrature points |

Implemented (filter by PDE, Heaviside projection, P0 and P1 design fields):

```julia
af(ρ̃, v) = ∫(r^2 * ∇(ρ̃) ⋅ ∇(v) + ρ̃ * v)dΩ
Sf = LinearOperator(StaticLinearModel(af, Uf, Vf, NothingBC(); ls = LUSolver()); depends = (),
                    solve = (m, b, ctx, stale) -> solve!(m, b; Assembly = stale))

z = OptVariables(ρ = FieldVariable(Vρ; bounds = (0.0, 1.0), init = 0.5))
x = DesignFields(z; ρ = Identity())
θ = PhysicalFields(x; ρ = Heaviside(β = 1.0, η = 0.5) ∘ HelmholtzFilter(Sf; measure = dΩ))

# the same in one line
z, x, θ = single_field_design(:ρ, Vρ; bounds = (0.0, 1.0), init = 0.5,
                              physical = Heaviside(β = 1.0, η = 0.5) ∘ HelmholtzFilter(Sf; measure = dΩ))
```

- Maps are composed with `∘` in the mathematical order: `H ∘ F` filters first and projects after.
- The filter output lives in the trial space of the filter model (usually P1), also for a P0 design field. The
  projection is applied point by point at the quadrature points, so `θ.ρ` is a Gridap `CellField`.
- `θ.ρ` is a fixed object updated in place: weak forms capture it once (checked with the lazy and the compiled
  backends of HyperFEM).
- Several physical fields can come from one design field (`E = :ρ => map`); their sensitivities are added.
- Sensitivities: for linear forms `δ -> ∫(… * δ)dΩ` with respect to each physical field, `design_gradient!` returns
  the derivative with respect to `z` (derivative of the projection inside the integrals, then the transposed filter,
  which is the same solve because the filter form is symmetric).
- Bounds, initial values and maps are mandatory; write `Identity()` when there is no transformation.
- Only HyperShape writes `x` and `θ`; the user reads them as Gridap fields.

Planned: global parameters as blocks (applied voltage, fibre angle), frozen regions and scaling in the map `z → x`,
physical fields that depend on several design fields, material interpolation in the map `x → θ`.

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
init(opt, z, oracle)               -> state      # memory and history
step!(opt, state, z, oracle, ctx)  -> StepInfo   # overwrites z with the next design
reset!(opt, state)                               # forget the history
shrink!(opt, state, factor)                      # roll back the last step, reduce the step size
```

The oracle is the only thing an optimiser may ask: value and derivative of the objective, value at a trial point
(forward problem only), bounds, the linear (volume) constraint and the inner product of the design metric.
Implemented: `OC` (optimality criteria, as in top88) and `SiMPL` (mirror descent with a latent variable, Armijo or
Bregman line search). `FunctionOracle` builds an oracle from plain functions:

```julia
using HyperShape.Optimizers
using HyperShape.Optimizers: init, step!

t = [0.9, 0.1, 0.7, 0.4]
oracle = FunctionOracle(z -> sum(abs2, z .- t) / 2, (dF, z) -> (dF .= z .- t);
                        lb = 0.0, ub = 1.0, constraint = (ones(4), 1.6))   # Σz ≤ 1.6
z = fill(0.4, 4)
opt = SiMPL()
state = init(opt, z, oracle)
for k in 1:50
  info = step!(opt, state, z, oracle, nothing)
  info.kkt < 1e-6 && break
end
```

## Post-processing

`OptPostProcessor(problem, driver; every, filepath)` calls the user-written `driver(post, ctx)` every `every` design
iterations, following the pattern of HyperFEM's `PostProcessor`. The user decides which fields are written and on
which triangulation, including domains larger than the design region.
