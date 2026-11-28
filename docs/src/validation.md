# Validation

Linear validation of phase 0: compliance minimisation with SIMP (p = 3), a Helmholtz filter and a volume constraint on
the physical density, against published values and codes. The scripts are in the folder `validation/` of the repository; the
fast cases run in the test suite. Measured on an Apple M5 Max, Julia 1.13.0, 4 threads, `LUSolver`. Each script
also writes the fixed report table of its runs (`J_bin`, volumes, grey, perimeter, holes, solves, time); those tables,
the meaning of their columns and the rule for comparing methods are on the [Benchmarks](benchmarks.md) page.

## MBB beam against top88

Andreassen et al., *Efficient topology optimization in MATLAB using 88 lines of code*, SMO 43 (2011): volume 0.5,
E₀ = 1, Emin = 1e-9, ν = 0.3, plane stress, unit square elements, rmin = 0.04 nelx, unit point load at the top-left
node, OC with move 0.2, stop at max|Δx| ≤ 0.01. The paper publishes the compliances of the final designs in its figure
captions: Fig. 3, the density filter of the code (`ft = 2`, a convolution filter): 233.71 (60×20) and 235.73
(150×50); Fig. 4, the PDE (Helmholtz) density filter of section 4.2: 237.60 and 235.36. Two comparisons
(`validation/mbb_top88.jl`):

- **Against Fig. 4 (1 % on J).** HyperShape's P0 design with the cell mean of the filtered field and OC as top88 is
  the formulation of section 4.2 on the same discretisation (its filter uses `r = rmin/(2√3)`, as HyperShape's
  `HelmholtzFilter(dΩ; rmin)`): it gives 237.606 and 235.363, +0.00 % on both meshes.
- **Against top88.m with its density filter (5 % on `J_bin`).** The code `top88.m` was run in Octave 9.1 with
  `ft = 2` (`validation/data/top88`, CSV only: the code is not included); its last printed compliances, 233.7146 and
  235.7337 (144 and 362 iterations), are those of Fig. 3. Another filter on the same mesh, elements, load and supports:
  the rule between methods, on `J_bin` (the design read as 0/1 per cell, evaluated with the same finite elements for
  both; J of top88 below is its final design evaluated with the finite elements of HyperShape's problem, which are
  those of top88).

| Case | J | vs Fig. 4 | `J_bin` | `J_bin` vs top88.m | iterations | time |
|---|---|---|---|---|---|---|
| 60×20 Andreassen et al. (2011), Fig. 4, PDE filter | 237.60 | — | — | — | — | — |
| 60×20 top88.m in Octave (density filter, ft = 2; Fig. 3: 233.71) | 233.713 | — | 187.446 | — | 144 | — |
| 60×20 P0, cell mean, OC as top88 | 237.606 | +0.00 % | 187.333 | −0.06 % | 138 | 1.4 s |
| 60×20 P0, Gauss points, OC as top88 | 235.032 | — | 188.342 | +0.48 % | 88 | 0.9 s |
| 60×20 P1, Gauss points, OC as top88 | 237.555 | — | 187.347 | −0.05 % | 137 | 1.3 s |
| 60×20 P0, cell mean, `OC()` | 238.978 | — | 188.369 | +0.49 % | 72 | 0.7 s |
| 60×20 P0, Gauss points, `SiMPL()` | 235.069 | — | 188.056 | +0.33 % | 39 | 0.4 s |
| 60×20 P1, Gauss points, `SiMPL()` | 239.291 | — | 187.242 | −0.11 % | 36 | 0.3 s |
| 150×50 Andreassen et al. (2011), Fig. 4, PDE filter | 235.36 | — | — | — | — | — |
| 150×50 top88.m in Octave (density filter, ft = 2; Fig. 3: 235.73) | 235.735 | — | 189.651 | — | 362 | — |
| 150×50 P0, cell mean, OC as top88 | 235.363 | +0.00 % | 190.528 | +0.46 % | 457 | 18.7 s |
| 150×50 P0, Gauss points, OC as top88 | 234.75 | — | 190.473 | +0.43 % | 543 | 21.4 s |
| 150×50 P1, Gauss points, OC as top88 | 235.373 | — | 190.21 | +0.29 % | 365 | 15.0 s |
| 150×50 P0, cell mean, `OC()` | 237.021 | — | 191.381 | +0.91 % | 58 | 2.4 s |
| 150×50 P0, Gauss points, `SiMPL()` | 235.542 | — | 190.702 | +0.55 % | 45 | 2.2 s |
| 150×50 P1, Gauss points, `SiMPL()` | 235.934 | — | 191.007 | +0.72 % | 48 | 2.2 s |

*OC as top88* is `OC(adapt = nothing)` with `StopCriteria(change_tol = 0.01)`; `OC()` and `SiMPL()` use the defaults of
HyperShape (adaptive move limits, stop when the objective settles). The script also writes a figure per mesh with
the design of top88 beside that of HyperShape (P0, cell mean, OC as top88).

- **The two filters.** Against top88.m with its convolution filter the J of the grey designs differs by up to 2.4 %
  (+1.67 % for the formulation of top88 with the cell mean on 60×20); read as 0/1 the designs are within 0.5 % on
  60×20 and 0.9 % on 150×50. Most of the difference in J is the equivalence of the radii: `r = rmin/(2√3)` matches
  the second moment of the cone filter in 1D (the convention of Lazarov and Sigmund 2011 and of section 4.2); in 2D the
  second moments match with r ≈ 0.274 rmin, 5 % smaller, and with that radius the cell mean on 60×20 gives J = 234.76
  (+0.45 %), grey 0.335 (top88 0.330) and `J_bin` +0.09 % (measured in the review of the bank, 2026-11-27). `J_bin`
  hardly depends on it. The test suite keeps 237.60 (Fig. 4) and checks the design of top88.m against Fig. 3.
- **Cell mean or Gauss points.** top88 gives each element one physical density, the cell mean of the filtered nodal
  field. HyperShape evaluates the filtered field at the Gauss points, so the stiffness varies inside an element: a
  different formulation, 1.1 % stiffer at 60×20 and 0.3 % at 150×50. The cell mean is obtained by composing
  [`CellMean`](@ref) after the filter, `CellMean(dΩ) ∘ HelmholtzFilter(dΩ; rmin, ls)`.
- **Point load.** It is written as the traction dual to the shape function of the corner node on the first top edge,
  `t(x) = −(4 − 6x/h)/h` for x < h: its work on a P1 field is the value at the corner.
- The filter has homogeneous Neumann conditions, as in top88 and SiMPL: the design sticks to the boundary of the domain.

## MBB beam against SiMPL

Kim, Lazarov, Surowiec, Keith, *A simple and efficient algorithm for topology optimization* (SiMPL),
arXiv:2411.19421v3. Table 1 (768×256): SiMPL 1.2078e-3 in 50 iterations, OC 1.2234e-3 after 300; table 2 (h = 1/256,
30 iterations): 1.0566e-3. Setup of the authors' code (problem MBB2 of `miniapps/simpl` in dohyun-cse/mfem):
(0, 3)×(0, 1), 768×256, plane strain, ν = 0.3, Emin = 1e-6, volume 0.3, P0 control, body force (0, −1) in the disc of
radius 0.05 at (0, 1), roller on the bottom edge for x > 3 − 1/32, stop at the stationarity measure
Sₖ = ‖z − P(z − g)‖ ≤ 1e-5 (L2 metric, `StopCriteria(stationarity_tol = 1e-5)` with `LumpedMass(dΩ)`).

| Case | C | vs reference | iterations | time |
|---|---|---|---|---|
| rmin 0.05, P0, `SiMPL()`, authors' sensitivity (table 1) | 1.2078e-3 | +0.00 % | 50 | 81.7 s |
| rmin 0.05, P0, `SiMPL()`, exact sensitivity | 1.1960e-3 | −0.97 % | 104 | 146.3 s |
| rmin 0.05, P1, `SiMPL()`, exact sensitivity | 1.2030e-3 | −0.40 % | 73 | 120.2 s |
| rmin 0.02, P0, `SiMPL()`, authors' sensitivity, 29 steps (table 2) | 1.0566e-3 | +0.00 % | 29 | 42.7 s |
| rmin 0.05, P0, `OC()`, default stop (vs OC of table 1) | 1.2113e-3 | −0.99 % | 102 | 151.7 s |

- **Filter radius.** Table 1 was computed with rmin = 0.05: the `run_local.sh` of dohyun-cse/mfem 28e17f2
  (2024-11-25, before the paper) holds the command of table 1 without `-fr`, and 0.05 was the default of MBB2 until
  commit c5b9f24 (2025-02-14), which changed it to 0.02. The commit given in the paper (022954b) only runs the mesh
  study of table 2, with 0.02.
- **Sensitivity of the authors' code.** It halves the shear term (`topopt.cpp`: `M = mu*0.5`, then
  `2.0*M*grad:adjgrad`), so it is the derivative of `λ tr(ε)² + μ ε:ε` instead of `λ tr(ε)² + 2μ ε:ε`: not the exact
  derivative of their stiffness. `mbb(…; sensitivity = :authors)` in `validation/common/problems.jl` replicates it. With it,
  HyperShape reproduces the iterates of the authors' log to eight digits (first iterate checked in the test suite at
  192×64) and both tables. With the exact derivative the same setup reaches a 0.97 % stiffer design, in more iterations.
- **Iterations.** The authors count the initial design as iteration 0, so their 30 iterations of table 2 are 29 steps.
- **OC.** The OC value of the paper is taken after 300 iterations without convergence; `OC()` with adaptive move
  limits and the default stop reaches −0.99 %.

## 3D cantilever

(0, 2)×(0, 1)×(0, 1), 24×12×12 hexahedra (12 168 displacement dofs), clamped at x = 0, downward traction on the bottom
strip (height h) of the face x = 2 (close to the line load of top3d), Emin = 1e-9, rmin = 2.5h, volume 0.3. No published value: `check_consistency`
passes for P0 and P1, and the design keeps the symmetry in y.

| Case | C | iterations | time | y-asymmetry |
|---|---|---|---|---|
| P0, `OC()` | 1.6581 | 82 | 14.3 s | 2.1e-12 |
| P0, `SiMPL()` | 1.2769 | 30 | 8.5 s | 1.1e-10 |
| P1, `OC()` | 1.7156 | 90 | 18.5 s | 3.8e-12 |
| P1, `SiMPL()` | 1.3149 | 30 | 8.5 s | 1.8e-9 |

OC and SiMPL end in different local minima, about 30 % apart: each one started from the design of the other stays
there (SiMPL from the OC design: 1.647; OC from the SiMPL design: 1.2749). Without continuation of the penalty the
result of a nonconvex problem depends on the path.

## Bridge of Takezawa et al. (double well)

Takezawa, Nishiwaki and Kitamura, *Shape and topology optimization based on the phase field method and sensitivity
analysis*, JCP 229 (2010), section 5.2: domain 2 × 1, roller at (0, 0), pin at (2, 0), unit point load at (1, 0), plane
stress (E = 1, ν = 0.3), volume 0.3, `k(φ) = max(φ³, 1e-5)`, κ = 1e-5 on a 200 × 100 grid, 100 iterations: published
J = 23.3. `ReactionDiffusion(; κ, reaction = DoubleWell())` (`τ = κ/|D|`) (the default scheme) and the preset
`:takezawa2010` (their explicit scheme), from their seed (rows of rectangular holes) and from `hole_seed` (`:hex`,
radius 0.1, spacing 0.4), 100 iterations (`validation/bridge_takezawa.jl`; 2026-11-26, `T = 25` per iteration: 5
substeps of 5 with the default scheme, 20 of 1.25 with the preset):

| method | seed | J | vs 23.3 | volume | interior holes |
|---|---|---|---|---|---|
| default | Takezawa | 23.668 | +1.58 % | 0.30002 | 9 |
| default | `:hex` | 23.899 | +2.57 % | 0.30002 | 6 |
| `:takezawa2010` | Takezawa | 23.614 | +1.35 % | 0.30008 | 9 |
| `:takezawa2010` | `:hex` | 23.661 | +1.55 % | 0.30007 | 8 |

The published value is met within 3 % in the four cases (the published design has 5–6 holes). The designs are
symmetric about x = 1 up to a few cells (0–2 % of the cells change phase in the mirror): rounding breaks the symmetry
and the evolution amplifies it. The seed of Takezawa et al. is used only here, against the published value on their
mesh; the examples and the tests start from `:hex`.

- **Mesh.** The load is a point load: the compliance grows like `log(1/h)` (the solid domain alone gives a larger J on
  a finer mesh), so the values of the bridge on different meshes are not comparable with each other nor with 23.3 away
  from 200 × 100. With κ = 1e-5 and `T = 25` on 400 × 200 (20 substeps of 1.25), the default stop gives J = 24.08 from
  the seed of Takezawa (175/185, settled/stopped, 9 holes; `interior_holes` gives 11 at 185, counting two grey specks
  of one and two cells next to the supports, χ ≈ 0.36) and 24.31 from `:hex` (141/142, 6 holes). The test suite runs
  the bridge on 100 × 50 with κ = 4e-5 from `:hex` against its own reference (J 23.34 at 100 iterations, 23.40 with the
  preset). The mesh independence is checked on the cantilever (next section).
- **Stop and settling.** With the default stop of the well (`material_tol = 1e-2`, maxiter 300, checked from the
  iteration after the settling): on 200 × 100 from the seed of Takezawa the run settles at 72 and stops at 82 (J 23.672, 9 holes), from `:hex` at 74 and 75 (J 23.902, 6 holes). 100 × 50 (κ = 4e-5, `:hex`): settles at 105, stops at 116 (J 23.263). The
  range of the objective alone (`StopCriteria(objective_range = 1e-3)`, not held) is met at the iteration of the
  settling, which checks the same range: from the seed of Takezawa it stops at 72 on 200 × 100 (J 23.719, 9 holes) and
  at 175 on 400 × 200 (J 24.117, 9 holes). With the floor of `objective_range` described below it stopped early, at 51
  (J 23.776) and at 50 (J 25.241, 21 holes: before the merge of the holes) (measured on the code before the
  correction). On 400 × 200 with 10 substeps of 2.5 from the seed of Takezawa the run used to stop at
  iteration 53 (J 25.21, 18 holes) with `settle = false` or with the stop of the well given by the user (not held).
  That stop was false, and not caused by the substeps: the relative range of the objective over the last 10
  iterations was 3.5e-3, above `1e-3`, but `objective_range` then compared the range with `max(max|f|, 1e-3 |f₁|)`
  and `f₁ = 3.16e5` (the holes of the seed over the supports), a floor of 315 for `J ≈ 25`. That floor has been
  removed: the range is relative to the window alone (`objective_atol` gives an absolute tolerance for objectives that
  tend to zero). It also caused the stops at 77 (200 × 100 above) and 93 (this case with the hold) documented before.
  Measured without it (2026-11-26, Mac M5 Max, Julia 1.13, 4 threads): this case settles at 88 and stops at 98
  (J 24.83, still 18 holes: on that mesh 10 substeps of 2.5 do not merge the holes of that seed; with `T = 25` from
  `n = ⌈T/(cfl Δt_lim)⌉`, 20 substeps of 1.25, it has 9 holes), the same with the stop given by the user; with
  `settle = false` the default stop is not met in 300 iterations (J 24.64).

## Mesh independence of the double well (cantilever)

The cantilever 2 × 1 (clamped at x = 0, a unit traction on x = 2, |y − ½| ≤ 0.1, volume 0.5, `k(φ) = max(φ³, 1e-5)`)
with `ReactionDiffusion(; κ, reaction = DoubleWell())`, κ = 1e-5 and `T = 25` (physical, the same on every
mesh), from `hole_seed` (`:hex`, radius 0.1, spacing 0.4) and the default stop (`validation/mesh_double_well.jl`;
2026-11-26, Mac M5 Max, Julia 1.13, 4 threads):

| mesh | substeps × Δt | J | settled / stopped | interior holes | time |
|---|---|---|---|---|---|
| 100 × 50 | 2 × 12.5 | 2.5582 | 89 / 92 | 6 | 2.5 s |
| 200 × 100 | 5 × 5 | 2.4933 | 74 / 75 | 8 | 7.4 s |
| 400 × 200 | 20 × 1.25 | 2.4255 | 95 / 100 | 14 | 47.6 s |

J is within ±2.7 % of the value on 200 × 100 (V1 from a solid start: 2.380–2.384 on the same meshes), but the topology
depends on the mesh and on the seed: the double well does not nucleate holes, and the holes of the seed are resolved
and merged differently on each mesh (6, 8 and 14 interior holes). The substeps grow like `(h₀/h)²`: the pointwise
reaction needs `√(κ Δt) ≲ 0.7 h`, a limit like a CFL condition (80 substeps on 800 × 400). With κ scaled with `h²`
instead (the same interface in cells; discarded) the boundary moves a fixed number of cells per iteration and a finer
mesh does not finish the bridge in 100 iterations. The test suite runs the 100 × 50 case with κ = 4e-5 (5 substeps of
5, J 2.537).

## Cantilever of levelset88 (reaction–diffusion of a level set)

Otomori, Yamada, Izui and Nishiwaki, *Matlab code for a level set-based topology optimization method using a reaction
diffusion equation*, SMO 51 (2015): the cantilever of their code, 80 × 64 unit elements, clamped on the left side,
five unit nodal forces in the middle of the right side, plane stress (E = 1, Emin = 1e-4, ν = 0.3), volume 0.5,
τ = 2e-4. The reference is their code run in Octave 9.1 (`validation/data/levelset88`; the code is not included).
`ReactionDiffusion` is run as it is, with the parameters the paper fixes (`Δt = 0.1`, consistent mass,
`phi_bc = :zero_boundary`, `void_weight = 1e-4`, the volume lowered over 100 iterations, `ExactVolume(max_step =
0.005)`) and the defaults of the method for the rest (`validation/otomori_levelset88.jl`; 2026-11-27): a level set
through `Ersatz()`, `sensitivity = :physical` (the exact derivative of the compliance weighted by the material,
instead of their topological derivative of the spherical hole), the exact multiplier of the volume (instead of their
exponential rule), `L1Mean()` and the default stop (instead of their criterion). A comparison of the method, not a
reproduction of the code. The compliance of the design of HyperShape is also evaluated as levelset88 reads a design
(binary elements, mean of φ over the element positive); the same evaluation of the Octave design gives its J within
5e-12.

| | stop | J | J as binary elements | volume (elements) | interior holes |
|---|---|---|---|---|---|
| levelset88 (Octave) | 112 (their criterion) | — | 534.95 | 0.4996 | 8 |
| `ReactionDiffusion` | 129 (the default stop; 128 iterations) | 519.40 (`Ersatz()`) | 532.54 (−0.45 %) | 0.4992 | 5 |
| `ReactionDiffusion` at iteration 112 | — | 521.75 | 535.43 (+0.09 %) | 0.4988 | — |

Iteration `k` of HyperShape is iteration `k + 1` of levelset88 (which records J before the update). The compliance
of the binary design ends within 0.5 % of levelset88, with a design of the same layout (two arcs and a central
joint) but fewer holes (5 against 8; 18 % of the elements in a different phase). During the descent of the volume the
two paths differ (up to 10 % in J around iteration 30, when levelset88 has removed more material than its target),
and the binary reading of the HyperShape design has a peak at iteration 101 (+9.7 %) that the cut cells of `Ersatz()`
do not see. levelset88 run on without its stop drifts (J 539 at 129, a jump to 585 near 185).
