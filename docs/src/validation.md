# Validation

Linear validation of phase 0: compliance minimisation with SIMP (p = 3), a Helmholtz filter and a volume constraint on
the physical density, against published values. The scripts are in the folder `validation/` of the repository; the
fast cases run in the test suite. Measured on an Apple M5 Max, Julia 1.13.0, 4 threads, `LUSolver`.

## MBB beam against top88 with the PDE filter

Andreassen et al., *Efficient topology optimization in MATLAB using 88 lines of code*, SMO 43 (2011), section 4.2:
volume 0.5, E₀ = 1, Emin = 1e-9, ν = 0.3, plane stress, unit square elements, rmin = 0.04 nelx, unit point load at the
top-left node, OC with move 0.2, stop at max|Δx| ≤ 0.01. Reference compliances 237.60 (60×20) and 235.36 (150×50).
A plain-Julia port of the code (`validation/top88.jl`) gives 237.5996 in 140 iterations and 235.3632 in 461 (the last
compliance the code prints; the tables below give the compliance of the final design).

| Case | C | vs reference | iterations | time |
|---|---|---|---|---|
| 60×20 P0, cell mean, OC as top88 | 237.606 | +0.00 % | 138 | 7.7 s |
| 60×20 P0, Gauss points, OC as top88 | 235.032 | −1.08 % | 88 | 1.3 s |
| 60×20 P1, Gauss points, OC as top88 | 237.555 | −0.02 % | 137 | 4.3 s |
| 60×20 P0, cell mean, `OC()` | 238.978 | +0.58 % | 72 | 1.6 s |
| 60×20 P0, Gauss points, `SiMPL()` | 235.069 | −1.07 % | 39 | 2.7 s |
| 60×20 P1, Gauss points, `SiMPL()` | 239.291 | +0.71 % | 36 | 2.4 s |
| 150×50 P0, cell mean, OC as top88 | 235.363 | +0.00 % | 457 | 16.1 s |
| 150×50 P0, Gauss points, OC as top88 | 234.750 | −0.26 % | 543 | 17.8 s |
| 150×50 P1, Gauss points, OC as top88 | 235.373 | +0.01 % | 365 | 12.2 s |
| 150×50 P0, cell mean, `OC()` | 237.021 | +0.71 % | 58 | 2.2 s |
| 150×50 P0, Gauss points, `SiMPL()` | 235.542 | +0.08 % | 45 | 1.9 s |
| 150×50 P1, Gauss points, `SiMPL()` | 235.934 | +0.24 % | 48 | 2.0 s |

*OC as top88* is `OC(adapt = nothing)` with `StopCriteria(change_tol = 0.01)`; `OC()` and `SiMPL()` use the defaults of
HyperShape (adaptive move limits, stop when the objective settles).

- **Cell mean or Gauss points.** top88 gives each element one physical density, the cell mean of the filtered nodal
  field (`xPhys = TF'·K⁻¹·TF·x`). HyperShape evaluates the filtered field at the Gauss points, so the stiffness varies
  inside an element: a different formulation, 1.1 % stiffer at 60×20 and 0.3 % at 150×50. The cell mean is obtained by composing
  [`CellMean`](@ref) after the filter, `CellMean(dΩ) ∘ HelmholtzFilter(dΩ; rmin, ls)`; with it HyperShape reproduces
  top88 to four digits.
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
  derivative of their stiffness. `mbb(…; sensitivity = :authors)` in `validation/cases.jl` replicates it. With it,
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
