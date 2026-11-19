# Design

!!! note
    Phase 0 (linear density methods) is implemented; the sections marked *planned* describe later phases. Names may
    still change until the package is registered.

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

Implemented (filter by PDE, tanh projection, P0 and P1 design fields):

```julia
F = HelmholtzFilter(dΩ; rmin = 0.05, ls = LUSolver())   # r = rmin/(2√3), P1, Neumann; ls goes to StaticLinearModel

z = OptVariables(ρ = FieldVariable(Vρ; bounds = (0.0, 1.0), init = 0.5))
x = DesignFields(z; ρ = Identity())
θ = PhysicalFields(x; ρ = TanhProjection(β = 1.0, η = 0.5) ∘ F)

# the same in one line
z, x, θ = single_field_design(:ρ, Vρ; bounds = (0.0, 1.0), init = 0.5, map = TanhProjection(β = 1.0, η = 0.5) ∘ F)

# any other filter: a LinearOperator over a model written by the user (symmetric form)
af(ρ̃, v) = ∫(r^2 * ∇(ρ̃) ⋅ ∇(v) + ρ̃ * v)dΩ + ∫(r * ρ̃ * v)dΓ                     # Robin term on Γ
Sf = LinearOperator(StaticLinearModel(af, Uf, Vf, NothingBC(); ls = LUSolver()))
θ = PhysicalFields(x; ρ = HelmholtzFilter(Sf, dΩ))
```

- Maps are composed with `∘` in the mathematical order: `proj ∘ F` filters first and projects after.
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
  `HelmholtzFilter(Sf, dΩ; backend)` and `PhysicalFields(x; …, backend)`. Both give the same vectors up to
  rounding. On 65 000 cells the compiled backend computes a design gradient 1.5 times faster than the lazy one with
  one thread and 5 to 7 times faster with four (`julia -t N`); on meshes of about a thousand cells both take well
  under a millisecond.
- Bounds, initial values and maps are mandatory; write `Identity()` when there is no transformation.
- Only HyperShape writes `x` and `θ`; the user reads them as Gridap fields.
- The filtered density is a P1 field and `θ` is evaluated at the quadrature points, so the stiffness varies inside an
  element. Codes such as top88 use one value per element, the cell mean of the filtered field:
  `CellMean(dΩ) ∘ F` gives that formulation (see the [Validation](validation.md) page).

**Frozen zones** (solid or void the optimisation cannot change: the material under a load, at a support, a skin)
are declared once, on the design space, as Dirichlet tags with their values:

```julia
add_tag_from_vertex_filter!(labels, "solid", model, x -> (x[1] <= 2 && x[2] >= ny - 2) || (x[1] >= nx - 2 && x[2] <= 2))
Vρ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2, dirichlet_tags = "solid")
z, x, θ = single_field_design(:ρ, Vρ; bounds = (0.0, 1.0), init = 0.5, frozen = 1.0, map = proj ∘ filter)
```

- Their values are not variables: `z` holds the free values only (the space with Dirichlet values is the affine map
  `x = P z + x₀`), and `x` has the frozen values in the zones.
- The zone is real material in the physical field: at the end of every chain `x → θ`, on the cells of the zones, `θ`
  is the image of the frozen value by the pointwise maps (projection, `PointwiseMap`), whatever the filter (which
  would otherwise dilute it: with a filter alone `θ` was 0.73–0.98 in small zones). The derivative vanishes there; the
  mask costs no solve and keeps a linear chain linear, so `Volume` over filters stays a linear constraint. The filter
  still sees the zone as solid, so the free material next to it attaches smoothly.
- A zone is made of whole cells (every Dirichlet value belongs to a cell whose values are all frozen); a line of nodes
  is an error: tag the cells next to it. A vertex tag (`add_tag_from_vertex_filter!`) serves P0 and P1 alike.
- `frozen` is a number, one number per tag, or a function of the coordinates; inside the bounds. Several zones with
  different values (solid and void) are fine. Periodic spaces work; spaces with other linear constraints (symmetries)
  are not supported yet.
- `max_fraction` of `Volume` is a fraction of the whole domain, the zones included. `frozen_indicator(x, :ρ)` gives
  the zones as a P0 field (to write them); `check_consistency` checks that θ equals the frozen value there; the banner
  shows the frozen cells of each block, and the checkpoint keeps their number.

Planned: global parameters as blocks (applied voltage, fibre angle), scaling and symmetries in the map `z → x`,
physical fields that depend on several design fields, material interpolation in the map `x → θ`.

## Constitutive wrappers (planned, phase 1)

The design will enter the physics through constitutive models that wrap HyperFEM models by aggregation and return, besides
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
- `ctx` is a read-only context updated by HyperShape: design iteration, attempt, continuation values and phase.
  `solve!` passes it to every solve function of the user; outside the loop (checks, scripts) `nothing` is accepted.
- The adjoint is declared in the state, with an `AdjointProblem`: who solves the transposed system and the forms
  `pᵀ ∂R/∂θ` of the state. The functionals that read the state only give their partial derivatives.

Implemented:

```julia
model_u = StaticLinearModel(l, a, U, V, bc; ls = LUSolver())          # forms and solver of the user; they capture θ
# adjoint: the transposed tangent on the space with homogeneous Dirichlet values (U0, bc0), matrix only
model_p = StaticLinearModel((p, w) -> a(w, p), U0, V, bc0; ls = LUSolver())
S = StateOperator(model_u, θ;                                           # default solve: solve!(m; Assembly = true)
      adjoint = AdjointProblem(model_p; sensitivity = (u, p) -> (δ -> ∫((dk ∘ θ.ρ) * ∇(u) ⋅ ∇(p) * δ)dΩ)))
```

For a linear and symmetric state, `AdjointProblem(SymmetricState(); sensitivity)` solves the adjoint with the matrix and
the factorisation of the state itself (the forward model must be re-assembled by its solve, as the default one does);
`AdjointProblem(; sensitivity)`, without an operator, when only self-adjoint functionals read the state. The adjoint
field is `get_adjoint(S)`. A nonlinear forward state gives its own `solve`, and returns the convergence
flag:

```julia
model_u = StaticNonlinearModel(res, jac, U, V, bc; nls)
uh = get_state(model_u)                                                 # captured by the transposed tangent
S = StateOperator(model_u, θ;
      solve   = (m, ctx) -> solve!(m; stepping = (nsteps = 10, maxbisec = 5), RestartState = false),
      adjoint = AdjointProblem(StaticLinearModel((p, w) -> jac(1.0)(uh, w, p), U0, V, bc0; ls = LUSolver());
                               sensitivity = (u, p) -> (δ -> ∫(…)dΩ)))
```

- `solve` returns the solution vector (taken as converged) or a tuple whose last element is the convergence flag,
  as `(x, flagconv)` from HyperFEM's nonlinear models. A failed solve does not throw: `is_converged(S)` is `false`,
  states that depend on it are not solved and the functionals return `NaN`.
- A state is solved again only when one of its dependencies changed; a failed solve is not repeated for the same
  design.
- The adjoint model is wrapped in a `LinearOperator` with the dependencies of `S` (extra ones with `depends`): its
  matrix is assembled again only when the state changed, so one factorisation serves every gradient of a design. Its
  trial space must have homogeneous Dirichlet values (checked).
- A state that depends on another state cannot be differentiated yet (the sensitivity through the upstream state
  would be missing): building a functional on it is an error.

## Functionals

One `Functional` per objective or constraint:

```julia
J = Functional(Jform, S; dJdu, dJdθ = nothing, scale)
#   Jform(u, θ)       scalar value
#   dJdu(u)           ∂J/∂u, a linear form w -> ∫(…)dΩ in the test function of the adjoint; or SelfAdjoint()
#   dJdθ(u, θ)        ∂J/∂θ, a linear form δ -> ∫(…)dΩ (only if J reads θ directly)
```

The value applies the forward operator `S`. The derivative is one reverse sweep: `dJdu` seeds the adjoint of `S`,
the adjoint is solved once, and `dJdθ − pᵀ ∂R/∂θ` is pulled back through the maps down to `z` (the signs are put by
HyperShape). θ is the physical fields `S` depends on (give `fields = θ` if it depends on none or on several).
Implemented, with `functional_value(J, ctx)`, `functional_gradient!(gz, J, ctx)` and
`functional_gradient!(gz, (J₁, …, Jₙ), w, ctx)` (the derivative of `Σ wᵢ Jᵢ` in one sweep: one adjoint solve per state
and one transposed solve per filter, whatever the number of functionals):

- The forms are `δ -> ∫(… * δ)dΩ`, or a named tuple of such functions when θ has several fields.
- `dJdu = SelfAdjoint()` when `∂J/∂u` is the load of a linear and symmetric state (the compliance): `p = u`, no
  adjoint solve; valid only with homogeneous Dirichlet values in the state (checked).
- Design-only functionals (volume): `Functional(Jform, θ; dJdθ, scale)` with `Jform(θ)` and `dJdθ(θ)`.
- `scale` is mandatory: a positive number, `:initial` (absolute value of the first evaluation) or `1.0`.

## Consistency checks

`check_consistency(J1, J2, …; ctx)` (or a named tuple of functionals) checks, at the current design, what the user
wrote and declared, and returns a report (`r.passed`, and one line per check, displayed as a table):

- the gradient of each functional, by a Taylor test (the error must fall with order 2 as the step is halved);
- the maps x → θ of each physical field alone, so that a failing gradient is located in the derivatives of the functional or of the state;
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
| `DensityMethod(optimizer; metric)` | density in [0, 1] | any `Optimizer` (`OC`, `SiMPL`) |
| `ReactionDiffusion` (phase 2) | level set in [−1, 1] | one linear solve, then clipping |
| `LevelSet` (phase 2) | signed distance | velocity extension, advection, reinitialisation |

Implemented: `Optimization` and `DensityMethod` with the common loop.

```julia
Vol = Volume(θ, dΩ; max_fraction = 0.5)   # physical (projected) density: nonlinear in z; Volume(x, dΩ; …) is linear
problem = Optimization(objective = J, constraints = (volume = Vol,))   # z is the one behind J
result  = solve!(problem, DensityMethod(SiMPL(); metric = LumpedMass(dΩ));
                 maxiter = 300, stop = StopCriteria(stationarity_rtol = 1e-3),
                 continuation = Continuation(β = proj => [1, 2, 4, 8, 16, 32]; every = 40),
                 checkpoint = Checkpoint(filepath = "results/ckpt", every = 10),
                 monitors = (grey = Grey(θ, dΩ),),
                 post = OptPostProcessor((post, ctx) -> write_step(post, Ω, "rho" => θ.ρ); every = 10,
                                         filepath = "results/opt"))
result.status, result.history.objective, result.history.constraints.volume, get_values(result.z)
```

`Volume(θ, dΩ; max_fraction)` bounds the fraction of the domain filled by the physical density, the one the weak
forms read; `Volume(x, dΩ; max_fraction)` that of the design
density. The Helmholtz filter preserves the volume, the tanh projection does not: on a 150×50 half MBB beam with
target 0.5, the design volume leaves the physical one at 0.64–0.68, the physical volume holds it at 0.5000.

- **Constraints:** any number, named, each a functional alone (`g ≤ 0`, as `Volume`) or a
  `Constraint(F; max, min, equal)` in the units of `F` (with `scale = :bound` the scale is taken from the bounds).
  The banner shows the bounds in the units of `F`; the history and the table, the scaled value `(F − r)/s` (`r` the
  upper bound, or the lower one if there is none).
  Each optimiser declares which constraints it treats itself (`handles`, `max_native`): OC and SiMPL treat one
  `g ≤ 0` that solves no state; the others go to the augmented Lagrangian (below). The
  oracle gives the optimiser, for each constraint `i`, its bounds and kind (`constraint_info`), its value and
  derivative, and `weighted_gradient!`, the derivative of `w₀ f + Σ wᵢ gᵢ` in one reverse sweep (one adjoint solve
  per state). For the constraint that OC or SiMPL treats, a linear one (`Volume` over identity or filter maps, or `linear = true`) is checked to be linear at
  the start and its vector `a` is computed once. A nonlinear one is linearised by the optimiser at every step and
  corrected with evaluations of the true constraint until it holds to `constraint_tol` (OC `1e-4`, SiMPL `1e-6`;
  SiMPL keeps every iterate feasible; at an infeasible design, after a continuation jump, it first takes restoration
  steps that must reduce the violation, which can take several steps or end the stage as stalled). Cost per step: one
  transposed filter for its derivative and one filter per correction (about 1.5–3), no forward solve.
- **Augmented Lagrangian:** `DensityMethod(SiMPL(); metric, lagrangian = AugmentedLagrangian(Bertsekas(μ0 = 1.0)))`,
  the default whenever a constraint is not native (a lower bound, an equality, an interval, one that reads a state, or
  more than the optimiser takes). The optimiser sees `f + Σ term(gᵢ)` over those constraints (Bertsekas' rule:
  `term = μ/2 (y − Π(y))² − λ²/(2μ)` with `y = g + λ/μ` and `Π` the projection onto the bounds), whose gradient is
  one weighted reverse sweep; the multipliers `λ ← μ (y − Π(y))` are updated after every accepted iteration and the
  optimiser keeps its step size (`objective_changed!`: SiMPL keeps Barzilai–Borwein, which the bench needs). The
  penalty is fixed by default (`growth = 1`): a growing penalty makes the subproblem ill-conditioned for first-order
  optimisers (5 of 7 Hock–Schittkowski problems fail with SiMPL). `native = (:volume,)` or `()` chooses the split;
  OC needs one native constraint and is heuristic with an objective of mixed sign. The run converges only with every
  constraint met to its tolerance (`constraint_tol` of the optimiser for the native, of the augmented Lagrangian for
  the others). History: `constraints`, `violations`, `multipliers` (`λ` for the augmented Lagrangian) and
  `penalties`; the checkpoint (format 2) keeps the split, `λ`, `μ` and the counters, so a resumed run continues the
  same multipliers.
- **Metric:** `LumpedMass(dΩ)` (cell volumes for P0, lumped mass for P1) or `Euclidean()`; mandatory for SiMPL, OC
  uses it only for the stationarity measure (default `Euclidean()`; give `LumpedMass(dΩ)` to compare it with SiMPL).
- **Rejection** (`on_failure`, default `RejectStep(shrink = 0.5, max_attempts = 5)`): if the forward problem does not converge at the new design, the loop restores the previous one,
  calls `shrink!` and tries again, at most `max_attempts` times; then it throws `OptimizationFailure` (checkpoint written).
  SiMPL handles failures inside its line search; if it finds no step, the current continuation stage ends (in the last stage, or without continuation, the result is `:stalled`).
- **Continuation:** explicit list of values per stage, a jump every `every` iterations (or when the optimiser finds no
  step); with `close_early = true` also when the stopping criteria are met. The optimiser forgets its memory at each
  jump; the optimisation only stops in the last stage.
- **Stopping:** `maxiter` is mandatory. `stop` defaults to `StopCriteria(objective_range = 1e-3)` for every optimiser
  (`default_stop`): the range of the objective over the last 10 iterations of the last stage, relative to the
  objective, below 0.1 %. With a tanh projection `change_tol` and `kkt_tol` are rarely reached; the objective settles
  (half MBB beam 150×50: OC with adaptive move limits stops at iteration 214, 0.4 % above its value at 500 iterations; SiMPL at 210, within 0.3 % of 350 iterations). Other criteria:
  `kkt_tol`, `kkt_rtol` (relative to the first iteration), `stationarity_tol`, `stationarity_rtol` (the measure
  `‖z − P(z − g)‖_M` of the SiMPL paper, computed by OC and SiMPL at the design before each step and shown in the
  column `S` of the table and the history), `change_tol`; `stop = nothing` runs `maxiter` iterations. A run that
  oscillates or whose physical volume keeps growing may still end at `maxiter`.
- **Grey monitor:** `solve!(…; monitors = (grey = Grey(θ, dΩ),))` adds the measure of grey `4/|Ω| ∫ρ(1 − ρ)` to the history, the
  table and the closing line. It is fixed mostly by β, so it is a measure of quality, not a stopping criterion.
- **Context:** `OptContext` (iteration, attempt, trial, phase, continuation values) reaches every solve function of
  the user; phases `:setup`, `:check`, `:optimization` and `:linesearch`.
- **Checkpoint:** `checkpoint.toml`, `z_<k>.bin` and `history_<k>.csv` in a folder; resume with
  `solve!(…; resume = "results/ckpt")` (the folder; or the result of `load_checkpoint`). OC saves its memory (adaptive move limits, correction shift; `memory`/`restore_memory!`) and resumes exactly; SiMPL restarts its step size. An `OC` continued by `continuation` gets its `move` back at the end of `solve!`.
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
(forward problem only), bounds, the constraint (its vector when it is linear; its value and derivative when it is
nonlinear, which needs no forward solve) and the inner product of the design metric.
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

- `write_step` writes with a `HyperShape.FieldWriter` (public, not exported, as it is meant to move to HyperFEM), one
  per name, built at the first call and closed when `solve!` returns,
  fails or is interrupted. The mesh is built once and is continuous (one point per vertex of the grid, not one per
  cell and vertex as `createvtk`); the values come straight from the degrees of freedom: P0 as cell data, P1/Q1 as
  point data, pointwise operations on them and on constants (`θ.ρ`, `E ∘ θ.ρ`, `1 - ρ`) and `extend_field`. Other
  fields (gradients, discontinuous spaces) are evaluated by Gridap at the vertices, with a warning.
- `format = :vtu` (default, any ParaView, a `.pvd` collection; each file carries the mesh) or `:vtkhdf` (one HDF5
  file with every iteration and the mesh written once; `using HDF5`; ParaView ≥ 5.12; the same fields at every
  iteration). `precision = Float32` halves the values; `async` (default with more than
  one thread) writes in another task, so the loop only waits for a copy of the values.
- `createvtk` stays available: `get_pvd(post)[ctx.iteration] = createvtk(Ω, file; cellfields = …)`.
- The `.pvd` collection (iteration as time) is written again after every entry, so ParaView can open it during the
  run and it stays valid after an interruption; a resumed run keeps the entries up to its checkpoint (VTKHDF files
  drop the later steps).
- `filepath/history.csv` holds the history of the run, written again at every call.
- No extra solves: the forward solve of the starting design is the one of the first iteration, done earlier.

Per step, three fields (P0, a projected P1 and a P1 vector), Apple M5 Max, Julia 1.13.0, 4 threads (minimum of 6
consecutive steps; VTU sizes include the mesh):

| | 150×50 | 40³ | 64³ |
|---|---|---|---|
| `createvtk` | 4.5 ms, 2.0 MB | 44 ms, 34 MB | 196 ms, 139 MB |
| VTU | 0.28 ms, 0.76 MB | 1.3 ms, 8.6 MB | 5.5 ms, 35 MB |
| VTKHDF, Float32 | 0.57 ms, 0.22 MB | 0.97 ms, 2.2 MB | 2.4 ms, 8.7 MB |
| loop blocked with `async` (VTKHDF, Float32) | 0.05 ms | 0.40 ms | 1.5 ms |

The first step builds the mesh and the maps (about 1 s at 64³). Planned: PVTU and VTKHDF with MPI, XDMF, GPU, quadratic
cells, and moving the writers to HyperFEM (HyperFEM issue #257).

## Defaults and how to replace them

A default never removes generality: everything it sets can be replaced from the user's script. The banner of
`solve!` marks the defaults of the loop in use (stopping criteria, failure handling, metric).

| Default | Where | How to replace it |
|---|---|---|
| `solve!(m; Assembly = true)` | `StateOperator` of a `StaticLinearModel` | `solve = (m, ctx) -> …` |
| `solve!(m, b; Assembly = stale)` | `LinearOperator` of a `StaticLinearModel` | `solve = (m, b, ctx, stale) -> …` |
| `()` (constant matrix) | dependencies of a `LinearOperator` | `LinearOperator(model, deps)` |
| `RejectStep(shrink = 0.5, max_attempts = 5)` | `solve!(…; on_failure)` | `on_failure = RejectStep(…)` |
| `default_stop(optimizer)` | `solve!(…; stop)` | `stop = StopCriteria(…)`, `stop = nothing` or `stop = (info, history, ctx) -> Bool` |
| `Euclidean()` (only for the stationarity measure) | `DensityMethod(OC())` | `metric = LumpedMass(dΩ)`, a vector of weights or an object with `metric_weights(m, z)` |
| maps `Identity`, `HelmholtzFilter`, `TanhProjection`, `CellMean` | `PhysicalFields`, `single_field_design` | `PointwiseMap(f, df; parameters)` |
| continued objects `TanhProjection` (β), `OC` (move) | `Continuation` | any `Parameter` read by the user's forms or maps |
| monitor `Grey` | `solve!(…; monitors)` | any object with `monitor_value(m)` |
| `initial = true`, `history = true`, `name = "optimization"` | `OptPostProcessor` | the keyword of the same name |
| `format = :vtu`, `precision = Float64`, `compress = false`, `async = Threads.nthreads() > 1` | `OptPostProcessor` (writers of `write_step`) | the keyword of the same name; `write_step(post, w::HyperShape.FieldWriter, …)` with a writer of your own; or `createvtk` in the driver |

The solver is always the one of the user's model. For example, with PETSc (GridapPETSc) build the model with
`ls = PETScLinearSolver()`; the default `solve` then calls it. Any other solve (a preconditioner rebuilt only every few
iterations, a warm start, a matrix-free solver) goes in the user's own `solve` function.

