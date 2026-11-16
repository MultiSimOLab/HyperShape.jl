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
Sf = LinearOperator(StaticLinearModel(af, Uf, Vf, NothingBC(); ls = LUSolver()))   # constant matrix, default solve

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
| Dependencies | invalidate the result (mandatory) | invalidate the matrix (default `()`) |
| `solve` function | `(m, ctx) -> …` | `(m, b, ctx, stale) -> …` |
| Default `solve` (`StaticLinearModel` only) | `solve!(m; Assembly = true)` | `solve!(m, b; Assembly = stale)` |

- Dependencies are given positionally (`StateOperator(model, θ)`) or with `depends =`; an operator solves again only
  when one of them has changed.
- HyperShape never chooses a solver: a default `solve` calls the solver of the user's model. Nonlinear, staggered and
  transient models need the user's `solve` (the load stepping and the restart are modelling decisions).
- `ctx` is a read-only context updated by HyperShape: design iteration, attempt, continuation values and phase. Until
  the common optimisation loop exists, whoever updates the fields passes it (`nothing` when the fields are built).
- The adjoint is another state operator; its forms are written by the user, typically the tangent of the forward
  problem with trial and test functions swapped.

Implemented:

```julia
model_u = StaticLinearModel(l, a, U, V, bc; ls = LUSolver())          # forms and solver of the user; they capture θ
S = StateOperator(model_u, θ)                                          # default solve: solve!(m; Assembly = true)
uh = get_state(S)                                                       # fixed object
# adjoint: a(w, p) = ∂J/∂u[w] for all w, on the space with homogeneous Dirichlet values (U0, bc0)
model_p = StaticLinearModel(l_adj, (p, w) -> a(w, p), U0, V, bc0; ls = LUSolver())   # captures θ and uh
A = AdjointOperator(S, model_p)                                         # depends on θ and S
```

A nonlinear forward state gives its own `solve`, and returns the convergence flag:

```julia
S = StateOperator(StaticNonlinearModel(res, jac, U, V, bc; nls), θ;
                  solve = (m, ctx) -> solve!(m; stepping = (nsteps = 10, maxbisec = 5), RestartState = false))
A = AdjointOperator(S, StaticLinearModel(l_adj, (p, w) -> jac(1.0)(get_state(S), w, p), U0, V, bc0; ls = LUSolver()))
```

- `solve` returns the solution vector (taken as converged) or a tuple whose last element is the convergence flag,
  as `(x, flagconv)` from HyperFEM's nonlinear models. A failed solve does not throw: `is_converged(S)` is `false`,
  states that depend on it are not solved and the functionals return `NaN`.
- A state is solved again only when one of its dependencies changed; a failed solve is not repeated for the same
  design.
- `AdjointOperator(S, model)` is a `StateOperator` whose dependencies are those of `S`, then `S` (extra ones can be
  added); it checks that the trial space has homogeneous Dirichlet values. The functional checks that the forward
  state depends on θ and the adjoint on θ and on the forward state.

## Functionals

One `Functional` per objective or constraint:

```julia
J = Functional(Jform, S; derivative = DJform, adjoint = A, scale)
#   Jform(u, θ)       scalar value
#   DJform(u, p, θ)   total derivative with respect to θ, as a linear form
```

The value applies the forward operator `S`; the derivative applies the adjoint operator `A` and assembles `DJform`.
θ is the physical fields `S` depends on (give `fields = θ` if it depends on none or on several).
HyperShape then applies the transposed maps down to `z`. Implemented, with `functional_value(J, ctx)` and
`functional_gradient!(gz, J, ctx)`:

- `derivative(u, p, θ)` returns `δ -> ∫(… * δ)dΩ`, or a named tuple of such functions when θ has several fields.
- `adjoint` is mandatory: self-adjoint functionals pass `adjoint = :self` (then `p = u`) and include the sign in
  `derivative`.
- Design-only functionals (volume): `Functional(Jform, θ; derivative = DJform, scale)` with `Jform(θ)` and
  `DJform(θ)`.
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

A method is an operator `z ↦ z⁺` over the functionals. All methods share one loop (evaluate, differentiate, step,
accept or reject, continuation, stopping, checkpoint); only the step is specific.

| Method | Design field | Step |
|---|---|---|
| `SIMP(optimizer; metric)` | density in [0, 1] | any `Optimizer` (`OC`, `SiMPL`) |
| `ReactionDiffusion` (phase 2) | level set in [−1, 1] | one linear solve, then clipping |
| `LevelSet` (phase 2) | signed distance | velocity extension, advection, reinitialisation |

Implemented: `Optimization` and `SIMP` with the common loop.

```julia
Vol = Volume(θ, dΩ; max_fraction = 0.5)   # physical (projected) density: nonlinear in z; Volume(x, dΩ; …) is linear
problem = Optimization(z; objective = J, constraints = Vol)
result  = solve!(problem, SIMP(SiMPL(); metric = LumpedMass(dΩ));
                 maxiter = 300, stop = StopCriteria(kkt = 1e-5),
                 schedule = Continuation(β = H => [1, 2, 4, 8, 16, 32]; every = 40),
                 checkpoint = Checkpoint("results/ckpt"; every = 10),
                 post = OptPostProcessor((post, ctx) -> write_step(post, Ω, "rho" => θ.ρ); every = 10,
                                         filepath = "results/opt"))
result.status, result.history.objective
```

`Volume(θ, dΩ; max_fraction)` bounds the fraction of the domain filled by the physical density, the one the weak
forms read; `Volume(x, dΩ; max_fraction)` that of the design
density. The Helmholtz filter preserves the volume, the Heaviside projection does not: on a 150×50 half MBB beam with
target 0.5, the design volume leaves the physical one at 0.64–0.68, the physical volume holds it at 0.5000.

- **Constraint:** in phase 0, at most one, a design-only functional (`Volume`, or a `Functional(J, θ; derivative,
  scale, linear)`). A linear one (`Volume` over identity or filter maps, or `linear = true`) is checked to be linear at
  the start and its vector `a` is computed once. A nonlinear one is linearised by the optimiser at every step and
  corrected with evaluations of the true constraint until it holds to `constraint_tol` (OC `1e-4`, SiMPL `1e-6`;
  SiMPL keeps every iterate feasible; at an infeasible design, after a continuation jump, it first takes restoration
  steps that must reduce the violation, which can take several steps or end the stage as stalled). Cost per step: one
  transposed filter for its derivative and one filter per correction (about 1.5–3), no forward solve.
- **Metric:** `LumpedMass(dΩ)` (cell volumes for P0, lumped mass for P1) or `Euclidean()`; mandatory for SiMPL, OC
  does not use it (default `Euclidean()`).
- **Rejection** (`on_failure`, default `RejectStep(shrink = 0.5, max = 5)`): if the forward problem does not converge at the new design, the loop restores the previous one,
  calls `shrink!` and tries again, at most `max` times; then it throws `OptimizationFailure` (checkpoint written).
  SiMPL handles failures inside its line search; if it finds no step, the current continuation stage ends (in the last stage, or without continuation, the result is `:stalled`).
- **Continuation:** explicit list of values per stage, a jump every `every` iterations (or when the optimiser finds no
  step); with `close_early = true` also when the stopping criteria are met. The optimiser forgets its memory at each
  jump; the optimisation only stops in the last stage.
- **Stopping:** `maxiter` is mandatory. `stop` defaults to `StopCriteria(objective = 1e-3)` for every optimiser
  (`default_stop`): the range of the objective over the last 10 iterations of the last stage, relative to the
  objective, below 0.1 %. With a Heaviside projection `change` and `kkt` are rarely reached; the objective settles
  (half MBB beam 150×50: OC with adaptive move limits stops at iteration 214, 0.4 % above its value at 500 iterations; SiMPL at 210, within 0.3 % of 350 iterations). Other criteria:
  `kkt`, `kkt_rel` (relative to the first iteration), `change`; `stop = nothing` runs `maxiter` iterations. A run that
  oscillates or whose physical volume keeps growing may still end at `maxiter`.
- **Grey monitor:** `solve!(…; grey = Grey(θ, dΩ))` adds the measure of grey `4/|Ω| ∫ρ(1 − ρ)` to the history, the
  table and the closing line. It is fixed mostly by β, so it is a measure of quality, not a stopping criterion.
- **Context:** `OptContext` (iteration, attempt, trial, phase, continuation values) reaches every solve function of
  the user; phases `:setup`, `:check`, `:optimization` and `:linesearch`.
- **Checkpoint:** `checkpoint.toml`, `z_<k>.bin` and `history_<k>.csv` in a folder; resume with
  `solve!(…; resume = load_checkpoint(path))`. OC saves its memory (adaptive move limits, correction shift; `memory`/`restore_memory!`) and resumes exactly; SiMPL restarts its step size. An `OC` continued by `schedule` gets its `move` back at the end of `solve!`.
- **Start:** a starting design that violates the constraint is scaled towards the lower bound (a linear constraint:
  on the entries it acts on, with equality; a nonlinear one: by regula falsi to `-1e-6 ≤ g ≤ 0`); the factor is in
  `result.start_scaling`. A resumed design is not touched, nor the design at a continuation jump.
- **Convergence with a nonlinear constraint:** the stopping criteria count only when the constraint also holds to
  `constraint_tol`.
- **Screen output:** `verbose = 2` (default) prints a banner, a row per iteration (objective, its relative change,
  constraint, optimality measure, change, step, multiplier, continued parameters, solves, time), the events and a
  closing line with the solves of the call; `1` omits the rows; `0` prints nothing. The output goes to `io`.
- **Failures:** `OptimizationFailure` carries the message, the history up to the last accepted iteration and its
  number; the history also records the multiplier of the linear constraint.
- **Solves per iteration:** OC one filter, one forward, one adjoint (none if self-adjoint) and one transposed filter;
  SiMPL one filter and one forward per line-search trial (usually one), one adjoint and one transposed filter. After
  a rejected step the previous design is not solved again.

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
Implemented: `OC` (optimality criteria of top88, with adaptive move limits per entry by default) and `SiMPL` (mirror descent with a latent variable, Armijo or
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

`OptPostProcessor(driver; every, filepath)` calls the user-written `driver(post, ctx)` at the starting design, every
`every` accepted iterations, at the end and after a failure, following the pattern of HyperFEM's `PostProcessor`.
It does not take the problem: `solve!` passes it (`post.problem`), so it can be built before or after it. The user
decides which fields are written and on which triangulation; `extend_field(f, Ω; outside)` takes a field of the design
zone to a larger domain with a fixed value outside.

```julia
post = OptPostProcessor(every = 10, filepath = "results/mbb") do post, ctx
  write_step(post, Ω, "rho" => extend_field(θ.ρ, Ω; outside = 1.0), "u" => get_state(S))
end
```

- `write_step` writes with `createvtk` (options such as `compress = true` pass through); any other file can be added
  with `get_pvd(post)[ctx.iteration] = file`.
- The `.pvd` collection (iteration as time) is written again after every entry, so ParaView can open it during the
  run and it stays valid after an interruption; a resumed run keeps the entries up to its checkpoint.
- `filepath/history.csv` holds the history of the run, written again at every call.
- No extra solves: the forward solve of the starting design is the one of the first iteration, done earlier.

Planned (`fast-output`): a writer with the mesh written once and the degrees of freedom written directly, VTKHDF,
asynchronous writing, MPI and GPU (see HyperFEM issue #257).

## Defaults and how to replace them

A default never removes generality: everything it sets can be replaced from the user's script. The banner of
`solve!` marks the defaults of the loop in use (stopping criteria, failure handling, metric).

| Default | Where | How to replace it |
|---|---|---|
| `solve!(m; Assembly = true)` | `StateOperator` / `AdjointOperator` of a `StaticLinearModel` | `solve = (m, ctx) -> …` |
| `solve!(m, b; Assembly = stale)` | `LinearOperator` of a `StaticLinearModel` | `solve = (m, b, ctx, stale) -> …` |
| `()` (constant matrix) | dependencies of a `LinearOperator` | `LinearOperator(model, deps)` |
| `RejectStep(shrink = 0.5, max = 5)` | `solve!(…; on_failure)` | `on_failure = RejectStep(…)` |
| `default_stop(optimizer)` | `solve!(…; stop)` | `stop = StopCriteria(…)` or `stop = nothing` |
| `Euclidean()` (unused) | `SIMP(OC())` | `metric = …` |
| `initial = true`, `history = true`, `name = "optimization"` | `OptPostProcessor` | the keyword of the same name |
| `createvtk` | `write_step` | its keywords (`compress = true`, …), or any writer in the driver with `get_pvd(post)[it] = file` |

The solver is always the one of the user's model. For example, with PETSc (GridapPETSc) build the model with
`ls = PETScLinearSolver()`; the default `solve` then calls it. Any other solve (a preconditioner rebuilt only every few
iterations, a warm start, a matrix-free solver) goes in the user's own `solve` function.

