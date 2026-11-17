# Validation

Linear validation of phase 0: compliance minimisation with SIMP (p = 3), a Helmholtz filter and a volume constraint on
the physical density, against published values. The scripts are in the folder `validation/` of the repository; the
fast cases run in the test suite. Measured on an Apple M5 Max, Julia 1.13.0, 4 threads, `LUSolver`.

## MBB beam against top88 with the PDE filter

Andreassen et al., *Efficient topology optimization in MATLAB using 88 lines of code*, SMO 43 (2011), section 4.2:
volume 0.5, E₀ = 1, Emin = 1e-9, ν = 0.3, plane stress, unit square elements, rmin = 0.04 nelx, unit point load at the
top-left node, OC with move 0.2, stop at max|Δx| ≤ 0.01. Reference compliances 237.60 (60×20) and 235.36 (150×50).
A plain-Julia port of the code (`validation/top88.jl`) gives 237.5996 in 140 iterations and 235.365 in 461.

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

*OC as top88* is `OC(adapt = nothing)` with `StopCriteria(change = 0.01)`; `OC()` and `SiMPL()` use the defaults of
HyperShape (adaptive move limits, stop when the objective settles).

- **Cell mean or Gauss points.** top88 gives each element one physical density, the cell mean of the filtered nodal
  field (`xPhys = TF'·K⁻¹·TF·x`). HyperShape evaluates the filtered field at the Gauss points, so the stiffness varies
  inside an element: a different formulation, about 1 % stiffer here. The cell mean is obtained by composing an L2
  projection onto P0 after the filter (a second `HelmholtzFilter` whose model is the P0 mass matrix, see
  `validation/cases.jl`); with it HyperShape reproduces top88 to four digits. A named map for it (`CellMean`) is planned.
- **Point load.** It is written as the traction dual to the shape function of the corner node on the first top edge,
  `t(x) = −(4 − 6x/h)/h` for x < h: its work on a P1 field is the value at the corner.
- The filter has homogeneous Neumann conditions, as in top88 and SiMPL: the design sticks to the boundary of the domain.

## MBB beam against SiMPL

Kim, Lazarov, Surowiec, Keith, *A simple and efficient algorithm for topology optimization* (SiMPL),
arXiv:2411.19421v3, table 1: SiMPL 1.2078e-3, OC 1.2234e-3. Setup of the authors' code (problem MBB2 of
`miniapps/simpl` in dohyun-cse/mfem): (0, 3)×(0, 1), 768×256, plane strain, ν = 0.3, Emin = 1e-6, volume 0.3,
P0 control, body force (0, −1) in the disc of radius 0.05 at (0, 1), roller on the bottom edge for x > 3 − 1/32.

| Case | C | vs reference | iterations | time |
|---|---|---|---|---|
| P0, `SiMPL()` (Armijo), `StopCriteria(kkt = 1e-5)` | 1.2097e-3 | +0.16 % | 33 | 38.6 s |
| P1, `SiMPL()`, `StopCriteria(kkt = 1e-5)` | 1.2132e-3 | +0.45 % | 32 | 43.4 s |
| P0, `OC()`, default stop (vs OC of the paper) | 1.2113e-3 | −0.99 % | 102 | 124.6 s |

- **Filter radius.** These values use rmin = 0.05, the default of the authors' code; 0.02 gives −12 %. The value in
  the text of the paper has not been checked against the PDF.
- **Stopping.** The paper stops at its L2 stationarity measure Sₖ ≤ 1e-5; HyperShape's `kkt` is a different measure,
  so the iteration counts are not comparable one to one (with `kkt = 1e-7` the compliance is 1 % lower after 300
  iterations).

## 3D cantilever

(0, 2)×(0, 1)×(0, 1), 24×12×12 hexahedra (12 168 displacement dofs), clamped at x = 0, downward traction on the bottom
strip of the face x = 2 (as top3d), Emin = 1e-9, rmin = 2.5h, volume 0.3. No published value: `check_consistency`
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
