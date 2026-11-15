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
- The vectors HyperShape assembles (load vector of the filter, sensitivities) use `backend = :compiled` by default
  (HyperFEMKernels: forms compiled once into threaded kernels) or `backend = :lazy` (Gridap), chosen in
  `HelmholtzFilter(Sf; measure, backend)` and `PhysicalFields(x; …, backend)`. Both give the same vectors up to
  rounding. On 65 000 cells the compiled backend computes a design gradient 1.5 times faster than the lazy one with
  one thread and 5 to 7 times faster with four (`julia -t N`); on meshes of about a thousand cells both take well
  under a millisecond.
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
- `ctx` is a read-only context updated by HyperShape: design iteration, attempt, continuation values and phase. Until
  the common optimisation loop exists, whoever updates the fields passes it (`nothing` when the fields are built).
- The adjoint is another state operator; its forms are written by the user, typically the tangent of the forward
  problem with trial and test functions swapped.

Implemented:

```julia
model_u = StaticLinearModel(l, a, U, V, bc; ls = LUSolver())          # forms and solver of the user; they capture θ
S = StateOperator(model_u; depends = θ, solve = (m, ctx) -> solve!(m; Assembly = true))
uh = get_state(S)                                                       # fixed object
# adjoint: a(w, p) = ∂J/∂u[w] for all w, on the space with homogeneous Dirichlet values (U0, bc0)
model_p = StaticLinearModel(l_adj, (p, w) -> a(w, p), U0, V, bc0; ls = LUSolver())   # captures θ and uh
A = StateOperator(model_p; depends = (θ, S), solve = (m, ctx) -> solve!(m; Assembly = true))
```

- `solve` returns the solution vector (taken as converged) or a tuple whose last element is the convergence flag,
  as `(x, flagconv)` from HyperFEM's nonlinear models. A failed solve does not throw: `is_converged(S)` is `false`,
  states that depend on it are not solved and the functionals return `NaN`.
- A state is solved again only when one of its dependencies changed; a failed solve is not repeated for the same
  design.
- The functional checks that the forward state depends on θ and the adjoint on θ and on the forward state, and that
  the trial space of the adjoint has homogeneous Dirichlet values.

## Functionals

One `Functional` per objective or constraint:

```julia
J = Functional(Jform, DJform, S, A, θ; scale)
#   Jform(u, θ)       scalar value
#   DJform(u, p, θ)   total derivative with respect to θ, as a linear form
```

The value applies the forward operator `S`; the derivative applies the adjoint operator `A` and assembles `DJform`.
HyperShape then applies the transposed maps down to `z`. Implemented, with `functional_value(J, ctx)` and
`functional_gradient!(gz, J, ctx)`:

- `DJform(u, p, θ)` returns `δ -> ∫(… * δ)dΩ`, or a named tuple of such functions when θ has several fields.
- Self-adjoint functionals pass `A = S` (then `p = u`) and include the sign in `DJform`.
- Design-only functionals (volume): `Functional(Jform, DJform, θ; scale)` with `Jform(θ)` and `DJform(θ)`.
- `scale` is mandatory: a positive number, `:initial` (absolute value of the first evaluation) or `1.0`.

## Consistency checks

`check_consistency(J1, J2, …; ctx)` (or a named tuple of functionals) checks, at the current design, what the user
wrote and declared, and returns a report (`r.passed`, and one line per check, displayed as a table):

- the gradient of each functional, by a Taylor test (the error must fall with order 2 as the step is halved);
- the maps x → θ of each physical field alone, so that a failing gradient is located in `DJform` or in the adjoint;
- the symmetry of each filter, with the user's solve;
- the declared dependencies: value and gradient at a new design, as usual and after forcing every reachable state
  and filter to assemble and solve again;
- optionally, the tangent of a state against its residual: `tangents = (S => (res, jac),)`.

z, the fields, the states and the scales of the functionals are restored on return. States shared by several
functionals are solved once per design: each forward state at most `steps + 4` times. A Taylor test where J does not
change along the direction (a saturated projection) is reported as not tested. Objects that no declared dependency
leads to cannot be checked.

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
