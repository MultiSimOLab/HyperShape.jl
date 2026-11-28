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
  is an error. A vertex tag (`add_tag_from_vertex_filter!`) serves P0 and P1 alike. To freeze the material under a
  load or at a support given as a boundary tag (a face, an edge, a vertex), in any mesh and in 3D,
  `tag_adjacent_cells!(labels, "load_zone", model, "load"; layers = 1)` tags the cells that contain it (and more
  layers around them); use `"load_zone"` in `dirichlet_tags`.
- `frozen` is a number, one number per tag, or a function of the coordinates (constant on each frozen cell); inside
  the bounds. Several zones with different values (solid and void) are fine if they do not touch: the value of a zone
  is one number per cell. Periodic spaces work; spaces with other linear constraints (symmetries)
  are not supported yet.
- `max_fraction` of `Volume` is a fraction of the whole domain, the zones included. `frozen_indicator(x, :ρ)` gives
  the zones as a P0 field (to write them); `check_consistency` checks that θ equals the frozen value there; the banner
  shows the frozen cells of each block, and the checkpoint keeps their number. With a zone, `θ.ρ` is an operation on
  fields: the fast `FieldWriter` writes it through Gridap's slower path (a warning says so).

**Interface maps** turn a level-set-like field `φ` (`φ ≥ 0` solid; the variable of reaction–diffusion) into the
material, as one more map of the chain `x → θ`; the weak forms integrate the design region with the measure that
HyperShape gives, so that a later treatment by cut cells changes the quadrature and not the forms:

```julia
z  = OptVariables(φ = FieldVariable(Vφ; bounds = (-1.0, 1.0), init = x -> …))   # Vφ: P1; init may be a function of x
x  = DesignFields(z; φ = Identity())
θ  = PhysicalFields(x; ρ = :φ => Ersatz() ∘ HelmholtzFilter(dΩ; rmin, ls = LUSolver()))   # or ρ = :φ => Step()
dΩd = design_measures(θ; degree = 2)        # dΩd.domain: a Measure; solid, void, interface reserved for cut cells
a(u, v) = ∫(… θ.ρ …)dΩd.domain
Vol = Volume(θ, dΩd; max_fraction = 0.4)
```

- `Step()`: χ = 1 where φ ≥ 0 at each quadrature point (the abrupt step of Li et al. 2023). A cut cell has the
  material of its quadrature points: a higher `degree` smooths the jumps of the objective but does not remove them
  (measured: with degree 2, 4 and 8 the objective did not change in 81, 52 and 10–30 of 120 small moves of an inclined
  interface). No derivative with respect to φ: `DensityMethod` refuses it when the run starts, `check_consistency`
  reports the derivative checks as not applicable (`–`); it is for methods that take the sensitivity with respect to
  the physical field (reaction–diffusion, next).
- `Ersatz()`: the classical ersatz material, the fraction of each cell where φ ≥ 0 (P0), with its exact derivative.
  Closed forms on triangles and tetrahedra (no cancellation, no allocations); quadrilaterals and hexahedra are split
  into 4 triangles and 24 tetrahedra around their centre (symmetric, about 6 times more accurate than splitting by a
  diagonal against the bilinear field). Exact volume with any quadrature for the subdivided interpolant; O(h²) for the
  geometry. Its input must be a nodal P1 scalar field (the design field or a filter output). The derivative lives on
  the cut cells only: a design without interface does not move and holes are not nucleated by it.
- Frozen zones: the material of a zone is the step of its value (φ = 1 gives 1), with both maps.
- Cost of `Ersatz()` (Mac M5 Max, one thread, measured by the design adviser and the reviewer): 65 536 quadrilaterals,
  0.8 ms to update and 1.0 ms to pull back; 82 944 tetrahedra, 0.6 and 0.55 ms; a hexahedron cut by the interface
  about 350 ns (24 tetrahedra), one without interface a few ns.
- Interface maps are for interface methods (reaction–diffusion, next): `DensityMethod` refuses `Step()` and
  `Ersatz()` when the run starts (a density method interpolates the density itself, SIMP or RAMP). In interface
  methods the starting design need not meet the constraint: the evolution takes it there.

**Starting designs with holes** (`hole_seed`): the free values of a design field with a pattern of holes, given to
`FieldVariable` as `init`:

```julia
φ0 = hole_seed(Vφ; radius = 0.05, spacing = 0.2, exclude = "clamp", connect = ("clamp", "load"))
z  = OptVariables(φ = FieldVariable(Vφ; bounds = (-1.0, 1.0), init = φ0, frozen = 1.0))
```

- Two uses: an optional perforated start for methods that nucleate holes (`ReactionDiffusion(…; sensitivity =
  :physical)`, as Yamada et al. 2010; studies of the dependence of the design on the start), and the mandatory start
  of methods that only move the boundary (the double-well reaction–diffusion, `ReactionDiffusion(…; reaction =
  DoubleWell())`, after Takezawa et al. 2010), whose holes are those of the seed (with `bounds = (0.0, 1.0)`).
- `radius`, `spacing` (default `4radius`), `margin` and `width` are lengths in the units of the mesh, so the start
  does not change when the mesh is refined. Patterns: `:hex` (default; staggered rows in 2D, face-centred cubic in
  3D, a hole at the centre of the box; `jitter ∈ [0, 1)` moves the centres at random without letting two holes
  touch), `:lattice` (square or cubic) and `:cvt` (a centroidal Voronoi tessellation of the region where holes may go,
  by Lloyd's relaxation: even spacing on any geometry and around the excluded zones; from the `:hex` centres, or from
  random points at least `spacing` apart with `rng`; there `spacing` is the mean distance from a centre to its
  nearest neighbour: the number of holes starts as that of `:hex` over the region and is corrected after measuring
  that distance, at most twice, so slender parts get fewer holes than their measure suggests; `nholes` fixes the
  number instead (rounded to whole orbits with a symmetry); so `:cvt` gives fewer holes than `:hex` with the same
  `spacing`, 46 against 73 on a 2 × 1 plate; neighbours are at least `min_distance` apart, by default `2radius + (spacing − 2radius)/2`: after
  each Lloyd step closer centres are pushed apart; the holes of `:cvt` are inside the region, those of `:hex` may be
  cut by the boundary). The non-nucleating methods of the literature start from
  periodic staggered patterns (Takezawa et al. 2010); with the double-well prototype a staggered seed gave the bridge
  of Takezawa within 1–2 % of the published compliance and symmetric designs, and purely random seeds (Poisson disks)
  up to 20 % worse on the cantilever, which is why random points are only a start for `:cvt`.
- `symmetry`: reflections `(point = p, normal = n)` and rotations of order `n` (`(point = p, order = n)` in 2D,
  `(point = p, axis = a, order = n)` in 3D), alone or in a tuple: the seed is built in a fundamental region of the
  group and replicated, exactly symmetric; every replica keeps the distance between holes and the margins. The
  geometry and the zones must be symmetric too (a warning says when they are not). With `:hex` and `:lattice` the
  symmetry must also map the lattice onto itself (`origin` moves it): otherwise the replicas meet along seams with
  gaps, and a warning says so.
- The holes keep away (`radius + margin`) from the frozen zones of the space (read from it) and from the tags in
  `exclude`; the solid of the seed must join the tags of `connect` (default `exclude`) through faces, or a warning
  names the tag left out. Holes that no value of the field falls into (a mesh too coarse for the radius) give a
  warning; none at all, an error.
- The value is a clipped distance (`lb` at the centre with `width = radius`, the level `(lb + ub)/2` on the edge of the
  hole, `ub` beyond `radius + width`), the minimum with `base`; with `bounds = (0.0, 1.0)` it is a density with 0.5 on
  the edge. P0 and P1, any mesh in 2D and 3D; linear cost (a few microseconds per value; `:cvt` times its at most 60
  Lloyd steps per relaxation, up to three relaxations). Sequential. Not periodic across periodic sides. With a
  design space on part of the model, the cells outside it count as solid for `connect`. An error when every value
  would be inside a hole, a warning above half (`base` does not count).

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
leads to cannot be checked. A physical field through `Step()` has no derivative: its gradient and chain checks are reported as not
applicable (`–`, they do not count in `passed`) and the dependency check compares the values only.

## Methods

A method is an operator `z ↦ z⁺` over the functionals. All methods share one loop (evaluate, differentiate, step,
accept or reject, continuation, stopping, checkpoint); only the step is specific.

| Method | Design field | Step |
|---|---|---|
| `DensityMethod(optimizer; metric)` | density in [0, 1] | any `Optimizer` (`OC`, `SiMPL`) |
| `ReactionDiffusion(; τ, sensitivity)` | level set in [−1, 1] through `Ersatz()` or `Step()`, or density in [0, 1] | one linear solve (factorised once), clipping, exact multiplier of the volume |
| `LevelSet` (phase 2) | signed distance | velocity extension, advection, reinitialisation |

Implemented: `Optimization`, `DensityMethod` and `ReactionDiffusion` with the common loop.

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
- **Stopping:** `maxiter` is mandatory. `stop` defaults to `StopCriteria(objective_range = 1e-3)` for OC and SiMPL
  (`default_stop`; `ReactionDiffusion`: material, objective and constraints, below): the range of the objective over the last 10 iterations of the last stage, relative to the
  largest `|f|` of that window, below 0.1 % (no reference to the first objective of the run, whose value may be far
  from the final one; an objective that tends to zero needs `objective_atol`, an absolute tolerance on the range). With a tanh projection `change_tol` and `kkt_tol` are rarely reached; the objective settles
  (half MBB beam 150×50: OC with adaptive move limits stops at iteration 214, 0.4 % above its value at 500 iterations; SiMPL at 210, within 0.3 % of 350 iterations). Other criteria:
  `kkt_tol`, `kkt_rtol` (relative to the first iteration), `stationarity_tol`, `stationarity_rtol` (the measure
  `‖z − P(z − g)‖_M` of the SiMPL paper, computed by OC and SiMPL at the design before each step and shown in the
  column `S` of the table and the history), `change_tol`; `stop = nothing` runs `maxiter` iterations. A run that
  oscillates or whose physical volume keeps growing may still end at `maxiter`.
- **Grey monitor:** `solve!(…; monitors = (grey = Grey(θ, dΩ),))` adds the measure of grey `4/|Ω| ∫ρ(1 − ρ)` to the history, the
  table and the closing line. It is fixed mostly by β, so it is a measure of quality, not a stopping criterion.
- **Context:** `OptContext` (iteration, attempt, trial, phase, continuation values, sensitivity mode) reaches every
  solve function of the user; phases `:setup`, `:check`, `:optimization` and `:linesearch`; `sensitivity` is
  `:physical` only while `ReactionDiffusion(…; sensitivity = :physical)` computes the sensitivity of its step.
- **Checkpoint:** `checkpoint.toml`, `z_<k>.bin` and `history_<k>.csv` in a folder; resume with
  `solve!(…; resume = "results/ckpt")` (the folder; or the result of `load_checkpoint`). OC (adaptive move limits, correction shift) and SiMPL (latent variable, Barzilai–Borwein memory, multiplier, shift) save their memory (`memory`/`restore_memory!`, a named tuple of numbers and design vectors), and the offsets of the linear constraints are kept (unless the user changed the bound or the scale), so a run resumed from an accepted iteration gives the designs of the uninterrupted one bit for bit (its first iteration repeats one forward solve). The state of a run is an `OptState` (`problem.state[]` after `solve!`; `resume = problem.state[]` continues without files); after an exception or an interruption the memory of the optimiser is not kept, and the resumed run restarts it. Checkpoints have format version 4 (version 3, without the column `step_solves` of the history, is read with those rows at 0; a resumed run says so). An `OC` continued by `continuation` gets its `move` back at the end of `solve!`.
- **Factories and discretisation:** `Optimization(model; design_setup, physics_setup)` builds the problem from two
  functions of the user: `design_setup(model) -> (; z, x)` and `physics_setup(model, x) -> (; objective, constraints,
  monitors, output, …)` (every name returned is kept in `problem.design` and `problem.physics`). What is continued or
  independent of the mesh (a `TanhProjection` of a `Continuation`, a `Parameter`, the optimiser) is created once, outside
  them; spaces, measures, fields, states, functionals, monitors and the output driver are created inside.
  `solve!(…; discretization)` says when the problem is built again: `FixedMesh()` (default) never;
  `HyperShape.Methods.RebuildCheck(at | every; what = :design | :physics)` on the same mesh, to check the factories
  (with linear states and direct solvers it is the run of `FixedMesh()` bit for bit, the counts of solves aside; at
  each rebuild the screen says whether the objective at the rebuilt problem is identical to the one of the history).
  A rebuild goes through the captured `OptState`, as a resume does; after it `solve!` checks that the factories built
  new variables and functionals, kept the blocks, constraints (names, bounds, kind), monitors and output, and that a
  continued `TanhProjection` is in the new maps. With factories, monitors and the output come from `physics_setup`
  (`OptPostProcessor(; every, filepath)` without a driver); the post-processor closes its writers at each rebuild and
  keeps one collection. `result.problem` is the last problem. Adaptive meshes (phase 4) are discretisations with a
  change of mesh and a transfer of the state. Example: `examples/mbb_factories.jl`.
- **Start:** a starting design that violates the constraint is scaled towards the lower bound (a linear constraint:
  on the entries it acts on, with equality; a nonlinear one: by regula falsi to `-1e-6 ≤ g ≤ 0`); the factor is in
  `result.start_scaling`. A resumed design is not touched, nor the design at a continuation jump.
- **Convergence with a nonlinear constraint:** the stopping criteria count only when the constraint also holds to
  `constraint_tol`.
- **Screen output:** `verbose = 2` (default) prints a banner, a row per iteration (objective, its relative change,
  constraint, optimality measure, change, step, multiplier, continued parameters, solves, time), the events and a
  closing line with the solves of the call; `1` omits the rows; `0` prints nothing. The output goes to `io`. For a
  method whose step solves linear systems of its own (`ReactionDiffusion`), the column of solves is `f/a/F/s` and the
  summary adds `step N`: the hook `HyperShape.step_solves(opt, state)` (accumulated since `init`, `nothing` by default),
  which the loop differences per iteration into `history.step_solves` (column 11 of the CSV), as it does with the
  forward, adjoint and filter solves. A density method has none: its table and summary do not change.
- **Failures:** `OptimizationFailure` carries the message, the history up to the last accepted iteration and its
  number; the history also records the multiplier of the linear constraint.
- **Solves per iteration:** OC one filter, one forward, one adjoint (none if self-adjoint) and one transposed filter;
  SiMPL one filter and one forward per line-search trial (usually one), one adjoint and one transposed filter. After
  a rejected step the previous design is not solved again.

### Reaction–diffusion

`ReactionDiffusion(; τ, sensitivity)` evolves one nodal P1 field `φ` by an implicit step of the reaction–diffusion
equation `∂φ/∂t = −C g − λ + τ ℓ² Δφ` (Yamada et al. 2010, Choi et al. 2011, Otomori et al. 2015, Kim et al. 2020):

```julia
z  = OptVariables(φ = FieldVariable(Vφ; bounds = (-1.0, 1.0), init = 1.0, frozen = 1.0))   # level set, solid start
x  = DesignFields(z; φ = Identity())
θ  = PhysicalFields(x; χ = :φ => Ersatz())          # material: the fraction of each cell where φ ≥ 0
dΩ = design_measures(θ; degree = 4)
# … the state with E(χ) = Emin + χ (1 − Emin), the compliance C and Vol = Volume(θ, dΩ; max_fraction = 0.5)
result = solve!(Optimization(z; objective = C, constraints = (volume = Vol,)),
                ReactionDiffusion(; τ = 2e-4, sensitivity = :physical); maxiter = 400)
```

The whole script is `examples/cantilever_rd.jl`. The same method evolves a density (`bounds = (0.0, 1.0)`,
`χ = :φ => Identity()` and SIMP in the weak forms).

- **Step:** `(mass/Δt + τ ℓ² K) δ = −τ ℓ² (K z + k_d) − C g`, `φ* = z + δ`, then `z = clamp(φ* − λ φ̌, lb, ub)`. `K` is the
  diffusion matrix (`∇φ⋅∇v`, or the bilinear form `diffusion = (φ, v) -> ∫(…)dΩ` of the user), `k_d` the frozen values
  in the free rows, `ℓ² = |D|^(2/d)` (`|D|` the measure of the design domain, so `τ` is dimensionless; in 3D this
  scaling is provisional), `mass` lumped (default) or consistent. HyperShape assembles `m`, `K` (and `M`) once on the
  design space with Gridap's assembler and factorises the operator once with `ls` (Gridap's `LinearSolver`,
  `LUSolver()` by default, shown in the banner); `solve = (A, b, x, ctx, stale) -> x` replaces it (PETSc, an iterative
  solver). The operator is factorised again only when `Δt` or `τ` change (a `Parameter` of a `Continuation`, a
  rejected step). Frozen zones of the design space stay as they are (Neumann conditions for φ elsewhere).
- **Volume** (`ExactVolume(; max_step, tol, maxiter, constraint)`, the default): the constraint built with `Volume`,
  wherever it is in the problem (with none or several, `solve!` asks for `ExactVolume(; constraint = :name)`); the
  multiplier `λ` is the root of
  `G(λ) = g(clamp(φ* − λ φ̌)) − g_k` with `φ̌ = A⁻¹ m` (`Δt` everywhere when no value is frozen): Newton on the
  clipped vector for a linear constraint (vectors only), Newton with the derivative of the constraint for `Ersatz()`,
  regula falsi (Illinois) for `Step()`, always inside the bracket given by the bounds, from the `λ` of the previous
  step. With an interface map the target goes down `max_step = 0.02` of `|D|` per iteration from the start (a solid
  start is not made feasible), with equality; at the final target the constraint is `g ≤ 0` with `λ ≥ 0`. A density
  meets the target from the first iteration. The augmented Lagrangian of the method takes every other constraint.
  The tolerance `tol = 1e-4` is a fraction of `|D|`, like `max_step` (converted with the scale of the constraint: the
  loop, the history and the table show the scaled value). With `Step()` the volume is piecewise constant: the root evaluates it with the material meter
  of the step (the quadrature of the `Volume` from the nodal values: exact, no constraint evaluation, no allocation)
  until at most one quadrature point changes side inside the bracket, and ends on its feasible side: the best
  feasible volume at the resolution of the quadrature points (checked against a sweep of λ). On coarse meshes the jump
  of one point exceeds `tol`. Measured on the 2 × 1 cantilever, 400 iterations (Mac of the lab, 4 threads, Julia
  1.13.0, 2026-11-24): `Ersatz()` 1.6 evaluations of the constraint per step at 120 × 60; `Step()` none through the
  constraint and 44, 31, 16, 4.2 and 3.8 evaluations of the meter at 20 × 10, 30 × 15, 40 × 20, 60 × 30 and 120 × 60;
  after the descent the volume of `Step()` stays within `tol` at 60 × 30 and 120 × 60 (0.2 and 0.7 cells below at
  most) and goes down to 1.2–3.5 cells below on 30 × 15 and 20 × 10, where the quadrature points of a cell change
  together (before: up to 20 cells below, with 3.5–3.7 evaluations of the constraint per step).
- **Sensitivity** (mandatory): `:physical` differentiates with respect to the material, the interface map taken as
  the identity weighted by the material, `(w₀ + (1 − w₀) χ) ∂j/∂χ` (`void_weight = w₀ = 1e-3`): holes nucleate where
  the material is little used. The exact derivative through the map would refill the void (in the prototype the
  design oscillated, J = 16–70 against 2.4). `:design` is the exact derivative with respect to φ: for a density
  (without an interface map both are the same) and with the double well (below), where it is the default.
  `:design` with an interface map is not a method and `solve!` throws: the exact derivative lives on the cut cells
  and, with the exact multiplier of the volume, which shifts φ uniformly, the solid plateau sinks where the derivative
  vanishes (measured from hole seeds on the 120 × 60 cantilever: 7–25 % worse than `:physical`, specks). The variant
  that only moves the boundary, from a start with holes (`hole_seed`), is a density with `reaction = DoubleWell(…)`. `check_consistency` always checks the exact derivative; with `Ersatz()` give it steps below the default,
  `check_consistency(…; h0 = 1e-3)`: larger steps cross the kinks of the cell fractions (a node changing sign) and
  the observed rates fall below 2 for a correct derivative.
- **Normalisation** (`normalize`): `L1Mean(c)` (default, `C = c |D| / Σ|gᵢ|`, independent of the mesh),
  `L2Norm()` (Kim et al.), `MaxAbs()`, or a function `(g, m, area) -> C`.
- **Stopping** (`default_stop`): the constraints met **and** the material changed in each of the last 5 iterations,
  `M_k = ∫|χ_k − χ_{k−1}| / ∫χ_k < 2e-3` (`χ` the field the `Volume` reads, cell by cell), **and** the relative range
  of the objective over the last 10 iterations below `1e-3`; the windows count only iterations with the constraints
  met, never the descent of the volume (`StopCriteria(material_tol = 2e-3, objective_range = 1e-3, combine = :all)`).
  `M_k` is computed by the step from the nodal values, without solves or allocations. Options:
  `StopCriteria(rate_tol = 0.1)` (`‖Δφ‖∞/Δt`), `change_tol` (`‖Δφ‖∞`) and `phase_tol` with `phase_window` (the lumped
  mass of the nodes where φ changes sign over `|D|`, only with an interface map).
- **Defaults of the evolution:** `Δt = 0.2` and `max_step = 0.02` of `|D|` per iteration with an interface map,
  `Δt = 0.4` and the target from the first iteration for a density (the banner says which). On the 2 × 1 cantilever
  they reach a stable topology in a few tens of iterations (study of 24 runs on 120 × 60 and 240 × 120 meshes). The
  topology of reaction–diffusion depends on the path: with a smaller `Δt` the design passes through metastable
  plateaus (9–16 holes, the objective and the material steady for 20 iterations or more) and the diffusion keeps
  merging holes long after the design looks converged; any reasonable criterion then stops on a plateau (a clean
  design, with an objective equal or better, but not the one of iteration 400). Neither the interface map (`Step()`
  or `Ersatz()`) nor the stopping criterion is a recommendation yet: the choice will be validated with more cases.
- **Diffusion and thickness:** `√(τ |D|)` (0.02 for `τ = 2e-4` on a 2 × 1 domain, 1.2 cells of a 120 × 60 mesh) is the
  length of a gradient (perimeter) energy, not a minimum thickness: bars of one or two cells are expensive but
  admissible, and they disappear only as the perimeter decreases, slowly. The 5th percentile of the thickness grows
  with `τ`, the minimum does not; a minimum size needs a filter or a constraint, which the method does not have.
- **Condition of the design field on the boundary** (`phi_bc`): `:neumann` (default, the natural condition of the
  diffusion), `:zero_boundary` (`φ = 0` on every node of the boundary of the design domain, found from the topology of
  its cells; Otomori et al. 2015) or `"tag" => value` pairs (`φ` held on the nodes of each tag, the nodes that
  `dirichlet_tags` would make Dirichlet; Yamada et al. 2010, Cui et al. 2023: `phi_bc = ("load" => 1.0,)`). The values
  are written into the design when `solve!` starts, before the first evaluation and any `scale = :initial`, and the
  step holds them in its operator: identity rows and columns, no increment and `φ̌ = 0` there, so the multiplier of the
  volume does not move them and the root of the volume leaves them out of its bracket (a value held at a bound would
  give 0/0). They stay values of `z`: not a frozen zone (a zone is a set of cells of material), and where a tag touches
  the frozen values of a zone the zone wins. One more back-substitution per factorisation (`φ̌`); not with the double
  well. With `:neumann` nothing of this runs: the runs are those of before bit for bit.
- **Rejection:** when the forward problem does not converge at the new design, `RejectStep` halves `Δt` (the operator
  is factorised again, the gradient comes from the snapshot of the oracle) and the nominal `Δt` comes back at the next
  step.
- **Checkpoint:** the multiplier is the memory of the step (`memory`) and the windows of the stopping criteria are
  saved too: a resumed run is the uninterrupted one bit for bit and stops at the same iteration.
- **Presets:** `ReactionDiffusion(:kim2020; τ, sensitivity)` sets the parameters of Kim et al. (2020): `L2Norm()`,
  consistent mass, the volume in the augmented Lagrangian with their rule (`Bertsekas(μ0 = 0.2, growth = 1.025,
  window = 1, progress = Inf, μmax = 2.0)`, `native = ()`); the keywords given replace those of the preset, and the
  banner marks them.
- **Per iteration:** one forward solve, one adjoint (none with `SelfAdjoint()`), one assembly of the sensitivity, one
  product `K z` and one substitution of the step, no factorisation; the volume costs a few evaluations (vectors for a
  density; the interface map and one integral for a level set). The solves of the step are in `history.step_solves`
  (one per step, one per implicit substep of the double well, none with the explicit scheme, plus the unit response
  of the exact volume at each factorisation when values are held), in the column `f/a/F/s` and in the summary
  (`step N`); the line below it adds the factorisations and the evaluations of the volume (through the constraint and
  by the material meter).
- **Coarse meshes with `Step()`:** the default stop is not met on the 20 × 10, 30 × 15 and 40 × 20 cantilevers in 400
  iterations (the material keeps changing by quadrature points, `M_k` of 1e-2 to 4e-2), while `Ersatz()` stops at 56,
  124 and 99 and the density at 35, 51 and 32; at 60 × 30 `Step()` stops at 123 and at 120 × 60 at 65. A run that
  reaches `maxiter` with the material still changing says so at the end. Without a meter (no `Volume` over the design
  domain read through `Identity()`, `Step()` or `Ersatz()`) the default stop uses the constraints and the objective
  range, and the banner says so.
- **Allocations:** the arithmetic of the step and a linear volume allocate nothing (one substitution of UMFPACK
  aside); each evaluation of a nonlinear volume reserves about 70 kB and each derivative of `Ersatz()` about 130 kB
  (Gridap's integration), independent of the mesh.
- **Not yet:** the presets of Otomori, Yamada and Choi, the second loop of
  Kim et al. (2020) (explicit double well with a decreasing diffusion, a refinement stage), the double well with an
  interface map, 3D validation.

### Double well

`ReactionDiffusion(; τ, reaction = DoubleWell())` is the variant that only moves the boundary (Takezawa, Nishiwaki and
Kitamura 2010; Kim et al. 2020, loop 2): a density `φ ∈ [0, 1]` read through `Identity()` (with an interpolation such as
`k(φ) = max(φ³, 1e-5)` in the weak forms), started from a design with holes, evolves by

```
∂φ/∂t = τ ℓ² Δφ − W w′(φ) − 30 w(φ) (η C g/m + λ),     w(φ) = φ² (1 − φ)²
```

The well keeps the design near 0/1 with an interface of width `ξ = √(τℓ²/2W)`; the sensitivity and the multiplier of
the volume are both weighted by `30 w(φ)`, which vanishes at 0 and 1, so the reaction moves only the interface and no
hole is nucleated (the start needs an interface: `solve!` throws otherwise and points to `hole_seed(Vφ; bounds =
(0.0, 1.0), …)`).

```julia
φ0 = hole_seed(Vφ; bounds = (0.0, 1.0), radius = 0.1, spacing = 0.4, exclude = ("roller", "pin", "load"))
z  = OptVariables(φ = FieldVariable(Vφ; bounds = (0.0, 1.0), init = φ0))
θ  = PhysicalFields(DesignFields(z; φ = Identity()); χ = :φ => Identity())
# … the state with k(φ) = max(φ³, 1e-5), the compliance C and Vol = Volume(θ, dΩ; max_fraction = 0.3)
result = solve!(Optimization(z; objective = C, constraints = (volume = Vol,)),
                ReactionDiffusion(; κ = 1e-5, reaction = DoubleWell()); maxiter = 300)   # τ = κ/|D| = 5e-6, T = 25
```

The whole script is `examples/bridge_dw.jl`; `validation/bridge_takezawa.jl` reproduces the published bridge.

- **Scheme** (decision of Jesús, 2026-11-25): `scheme = :implicit` (default) splits each of the `n` substeps of an
  iteration into an implicit diffusion with the operator of the step, factorised once,
  `(m/Δt + τℓ²K) ψ = m φ/Δt − τℓ² k_d`, and the semi-implicit pointwise reaction of Takezawa et al. (their Eqs. 44–45)
  applied to `ψ`, with `λ` inside the tilt term: the result stays in `[0, 1]` for any `Δt`. `scheme = :explicit` is
  their scheme (`ψ = φ − Δt τℓ² m⁻¹(K φ + k_d)`, no linear system). The sensitivity is computed once per iteration and
  frozen over the substeps.
- **Physical parameters** (decision of Jesús, 2026-11-26): `κ = τ ℓ²` and the time of the evolution per iteration
  `T` (`DoubleWell(T = 25)`) are physical, the same on every mesh. `ReactionDiffusion(; κ = …)` takes `κ` directly
  (exactly one of `τ` and `κ`; the banner shows both): `κ` is the coefficient of `K` in the operator as given, and
  `τ` enters as `τ ℓ²`, so `τ = κ/ℓ²` with the `ℓ²` of the method (a quadrature of `|D|`) gives the same run bit for
  bit (`τ = κ/2` on the domain 2 × 1 can differ in the last bits, and the double well amplifies them). With the well `κ` (or `τ`) and `Δt` are numbers: a
  `Continuation` of them would leave `n` and `Δt` stale, and `ReactionDiffusion` refuses a `Parameter` there. An iteration is `n = ⌈T/(cfl Δt_lim)⌉` substeps of
  `Δt = T/n`, with `Δt_lim = 2/maxᵢ(τℓ² Σⱼ|Kᵢⱼ|/mᵢ)` (Gershgorin, from the assembled `K` and `m` once at the start:
  `0.375 h²/(τℓ²)` on squares, any mesh) and `cfl = 4/3` with `:implicit`, `1/3` with `:explicit` (`h²/(8τℓ²)`, the
  step of Takezawa et al.). The pointwise reaction needs `√(τℓ² Δt) ≲ 0.7 h` (a limit like a CFL condition), so the
  number of substeps grows like `(h₀/h)²`: with κ = 1e-5 and `T = 25` on the domain 2 × 1, 2, 5, 20 and 80 substeps on
  100 × 50, 200 × 100, 400 × 200 and 800 × 400. No new factorisation per iteration. `DoubleWell(substeps = n)` fixes
  `n` instead (`T` is then ignored and `Δt = :auto` of the method is `cfl Δt_lim`; the preset `:takezawa2010`). The
  event after the setup gives `T`, `n`, `Δt`, `Δt_lim` and `ξ = √(τℓ²/2W)` in mean cell sizes `(|D|/cells)^(1/d)`
  (`K` and `m` are assembled once, in the setup; the banner gives the formula; once per `solve!`, and again after a
  rebuild of the discretisation only if it changed). More than 10⁶ substeps per iteration
  are refused. With `:explicit` the operator of the step is never factorised.
- **Mesh** (study of 2026-11-25, `validation/mesh_double_well.jl`): with κ = 1e-5 and `T = 25` the compliance of the
  cantilever 2 × 1 from `:hex` is the same within about 3 % on 100 × 50, 200 × 100 and 400 × 200 (2.558, 2.493, 2.426 with 2, 5 and 20 substeps: within ±2.7 % of the value on 200 × 100), but the
  topology depends on the seed and on the mesh (no holes are nucleated, and a coarse mesh resolves the seed and the
  thin bars differently). The bridge of Takezawa et al. has a point load: its compliance grows like `log(1/h)`, so its
  values on different meshes are not comparable (its validation is on 200 × 100, the mesh of the paper).
- **Volume:** the same `ExactVolume`; its multiplier acts inside the tilt (only on the interface, no `φ̌` and no extra
  solve): `G(λ) = g(R(ψ; η C g/m + λ)) − g_j` is non-increasing, its root is found at every substep on vectors (regula
  falsi, Illinois, from the previous `λ`), the target of the iteration is reached linearly over the substeps and
  `max_step = :auto` is `0.01` of `|D|` per iteration. A target out of reach in one substep (not enough interface)
  takes the end of the bracket; the next substeps catch up.
- **Defaults with the well:** `sensitivity` is `:design` (`:physical` is refused), `normalize = L2Norm()` (with
  `η = 10` it is `η = 20` of Takezawa et al. on their domain of measure 2), `W = 1/4`.
- **Settling** (decision of Jesús, 2026-11-25): with the sensitivity frozen over the substeps the interface overshoots
  and comes back, and 1–2 % of the material changes phase at every iteration. `settle = true` (default) goes to one
  substep (of the same `Δt`) per iteration when the constraints are met and the relative range of the objective over 10
  iterations is below `1e-3` (`settle = (window = 10, tol = 1e-3)`, both replaceable); an event says so. It depends
  only on the objective (no fraction of `maxiter`: a run resumed with a larger `maxiter` is the uninterrupted one), and
  a continuation jump starts the substeps and the window again. `DoubleWell(settle = false)` keeps the substeps.
- **Default stop with the well** (decisions of Jesús, 2026-11-25/26): after settling the interface still keeps a cycle
  of a few per mille of the material per iteration, so the default stop takes `material_tol = 1e-2` with the well
  (`2e-3` without it), the objective range `1e-3` over 10 iterations and the constraints unchanged. It is checked only
  from the iteration after the settling (or from the start with one substep, or with `settle = false`).
  `solve!(…; stop)` replaces it and is not held. The early stops of the study (400 × 200 and 800 × 400 from the seed
  of Takezawa, iterations 49–53) were not an interface stuck with the windows met: on 400 × 200 (iteration 53) the
  relative range of the objective over the window was 3.5e-3, above `1e-3`, but `objective_range` then measured it
  against `max(max|f|, 1e-3 |f₁|)`, and the first objective of those runs is huge (holes of the seed over the
  supports: `f₁ ≈ 3e5`, a floor of 315 for `J ≈ 25`). That floor, common to every method, has been removed: the range
  is relative to the window alone (`objective_atol` covers objectives that tend to zero). It also caused the stops at
  77 and 93 documented before (bridge 200 × 100 and the 400 × 200 case above with the hold, both from the seed of
  Takezawa). Measured (Validation page): the bridge 200 × 100 from `:hex` settles at 74 and stops at 75 (J 23.90; from
  the seed of Takezawa 72 and 82, J 23.67), the cantilever 200 × 100 at 74 and 75 (J 2.493).
- **Grey:** `StopCriteria(grey_tol)` stops on the measure of grey `Σᵢ mᵢ 4φ̂ᵢ(1 − φ̂ᵢ)/|D|` (nodal, without
  allocations); with the well it stays at 1–3 % (the interface), so it is a criterion for a refinement stage, not for
  the evolution from the seed.
- **Preset** `:takezawa2010`: `DoubleWell(W = 0.25, η = 20, scheme = :explicit, substeps = 20, settle = false)`, the
  normalisation `1/‖g‖_L²` (without `|D|`), `ExactVolume(max_step = 0.01)`; the script sets `τ = κ/ℓ²`, the seed and
  `maxiter = 100, stop = nothing`. Differences with the paper: an exact multiplier instead of their rule for ζ, P1
  finite elements with lumped mass instead of finite differences.
- **Per iteration:** one forward solve, one adjoint (none with `SelfAdjoint()`), `n` substitutions (none with
  `:explicit`) and `n` roots on vectors, no factorisation; the step allocates nothing but the substitutions of
  UMFPACK (nothing at all with `:explicit`).
- **Checkpoint:** the multiplier, the settling and the window of the objective that decides it are the memory of the
  step (the substeps follow from the settling and the setup): a resumed run is the uninterrupted one bit for bit,
  across the settling and with a larger `maxiter`.
- **Not offered:** the double well with an interface map (`Ersatz()`, `Step()`: φ ∈ [−1, 1] read through the map gave
  hundreds of specks in the prototype), `sensitivity = :physical` with the well, and the nucleation of holes.

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
| `ls = LUSolver()`, `mass = :lumped`, `normalize = L1Mean()`, `volume = ExactVolume()`, `Δt = :auto` (0.2 with an interface map, 0.4 without), `void_weight = 1e-3`, `lagrangian = AugmentedLagrangian()`, `phi_bc = :neumann` | `ReactionDiffusion` | the keyword of the same name; `solve = (A, b, x, ctx, stale) -> x` for any other solver; `diffusion = (φ, v) -> ∫(…)dΩ` |
| `max_step = :auto` (`0.02` of `|D|` with an interface map), `tol = 1e-4` (of `|D|`) | `ExactVolume` | the keyword of the same name |
| maps `Identity`, `HelmholtzFilter`, `TanhProjection`, `CellMean` | `PhysicalFields`, `single_field_design` | `PointwiseMap(f, df; parameters)` |
| continued objects `TanhProjection` (β), `OC` (move) | `Continuation` | any `Parameter` read by the user's forms or maps |
| monitor `Grey` | `solve!(…; monitors)` | any object with `monitor_value(m)` |
| `initial = true`, `history = true`, `name = "optimization"` | `OptPostProcessor` | the keyword of the same name |
| `format = :vtu`, `precision = Float64`, `compress = false`, `async = Threads.nthreads() > 1` | `OptPostProcessor` (writers of `write_step`) | the keyword of the same name; `write_step(post, w::HyperShape.FieldWriter, …)` with a writer of your own; or `createvtk` in the driver |

The solver is always the one of the user's model. For example, with PETSc (GridapPETSc) build the model with
`ls = PETScLinearSolver()`; the default `solve` then calls it. Any other solve (a preconditioner rebuilt only every few
iterations, a warm start, a matrix-free solver) goes in the user's own `solve` function.

