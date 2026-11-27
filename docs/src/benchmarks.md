# Benchmarks

The benchmark bank runs HyperShape on fixed problems with every method that applies and reports every run in one
fixed table, so that methods, discretisations and versions can be compared on the same footing. It lives in the
folder `validation/` of the repository, outside the package: the cases are scripts written with the public API, and
the shared helpers (`validation/common/`) are scripts included by them, not part of HyperShape. The
[Validation](validation.md) page discusses each case against its reference; this page describes the bank, its table
and the rule for comparing methods, and collects the tables of the cases that exist today.

## Running the bank

```
julia --project=validation -e 'using Pkg; Pkg.instantiate()'
julia --project=validation -t 4 validation/run_all.jl                    # every case, about TOTAL_MIN min
julia --project=validation -t 4 validation/run_all.jl mbb_top88 bridge_takezawa   # some of them
```

`run_all.jl` runs each case script in its own Julia process (with the threads of the calling one), keeps its output
in `results/<case>.log` and joins the tables of the cases into `results/report.csv` and `results/report.md`. Each
script can also be run alone (`julia --project=validation -t 4 validation/<case>.jl`); it prints its own table and
then writes its rows of the fixed table to `results/<case>.csv` and `results/<case>.md`. The folder is
`validation/results` (not versioned) or `ENV["HYPERSHAPE_RESULTS"]`. A case fails when its script stops with an
error or one of its checks does not hold (the script exits with code 1); `run_all.jl` reports it, still runs the
others and exits with code 1. Times are measured after runs on small meshes that compile every variant of the case,
with the machine free (one run at a time, load average checked before); the tables carry the machine, the threads,
the Julia version and the commit (`+dirty` when the tree had changes).

## The table

One row per run. A reference (a port of another code, a code run in Octave, a published value) has its own row, first,
with the measures of its design when its design is available, and "–" for what it does not have.

| column | meaning |
|---|---|
| case, variant | the problem and its mesh; the design variables, the method and the options that are not the defaults |
| J | the objective of the method on its final design (`result.objective`; the cases use `scale = 1.0`) |
| J\_bin | the objective of the final design read as 0/1 per cell: solid where the cell mean of the physical field is at least ½, with the same mesh, loads and supports and the material law `E = Emin + (1 − Emin) χ` (plain Gridap, `binary_evaluator`) |
| ref, vs ref | the reference value of the case and the deviation from it, with the column it is compared with (J or J\_bin) |
| V | the volume fraction that the constraint sees (`vf · (1 + g)`, from the history of the constraint `volume`, a `Volume(…; max_fraction = vf)` with its default scale) |
| V\_bin | the fraction of solid cells of the 0/1 design |
| grey | Sigmund's measure of grey of the cell means, mean of `4χ(1 − χ)` (0 for a 0/1 design; with a level set and `Ersatz()` it only sees the cut cells) |
| perimeter | of the 0/1 design, in domain units: the interior faces between a solid and a void cell times their length (an area in 3D). A staircase measure: comparable between methods on the same mesh; under refinement it tends to the ℓ¹ length of the boundary, not to the Euclidean one |
| holes | interior holes of the 0/1 design: connected components (by faces) of void cells that do not touch the boundary of the domain; in 3D, closed cavities. Specks of one or two grey cells count as holes |
| iterations | accepted iterations, with the status and the reason of the stop |
| solves | solves of the state, of the adjoint and of the filter (forward and transpose), summed over the history; and of the step of reaction–diffusion: not yet in the public history, so "–" (it will come with the counter of the step solves) |
| time | seconds of `solve!`, without compilation |
| time ÷ state solves | the time divided by the number of state solves: the cost of the method per state solve on that mesh and machine, with everything else included (filter, the step of reaction–diffusion, monitors such as the binary compliance of levelset88). It is not the time of one solve, and until the step solves are counted it does not compare reaction–diffusion with density methods |

## Comparing designs

The objective that a method optimises is not comparable between methods: SIMP with a filter keeps grey material,
a level set with `Ersatz()` has cut cells, a density of reaction–diffusion has a diffuse interface. The common
measure is `J_bin`, the same problem evaluated with the design read as 0/1 per cell. The tolerances of the bank are:

- **against another code with the same discretisation and formulation** (a published value of it or a port): J
  within **1 %**. Cases: the MBB with the cell mean and OC as top88 against Fig. 4 of Andreassen et al. (2011), the
  PDE filter (+0.00 %); SiMPL with the sensitivity of the authors' code against their tables 1 and 2 (+0.00 %);
- **against a published value** obtained with another discretisation or formulation: J within **5 %** (SiMPL with
  the exact sensitivity and OC: −0.40 % to −0.99 %). The bridge of Takezawa et al. uses 3 %, the tolerance decided for
  the double well (the default reproduces 23.3 within 2.6 %);
- **between methods** of HyperShape on the same mesh, and against another code or method run on the same mesh
  (top88.m with its convolution density filter, `ft = 2`, against the Helmholtz filter with `r = rmin/(2√3)`;
  levelset88): `J_bin` within **5 %**.

Each script checks its own rule and exits with code 1 when it does not hold (the 3D cantilever and the mesh study of
the double well have no reference value: their checks are consistency, symmetry, convergence and volume, and J in
the measured intervals).

## Cases

The cases that exist today, with their rows of the fixed table (`LUSolver`, backend `:compiled`; the discussion of
each case is on the [Validation](validation.md) page). The first row of the top88 and levelset88 tables is the
reference code; "–" marks what a row does not have. The SVG figures of the designs beside their reference
(`mbb_top88_<mesh>.svg`, `otomori_levelset88.svg`) are written to the results folder.

### MBB beam against top88 (Octave, density filter)

Rule between methods on `J_bin` (two filters, same mesh); reference: `top88.m` run in Octave 9.1, `ft = 2`, with `r = rmin/(2√3)` for the Helmholtz filter of HyperShape. `validation/mbb_top88.jl`; Apple M5 Max, 4 threads, Julia 1.13.0, 2026-11-27 19:48.

| case | variant | J | J_bin | ref | vs ref | V | V_bin | grey | perimeter | holes | iterations | solves state/adjoint/filter/step | time (s) | time / state solve (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MBB top88 60×20 | top88.m in Octave 9.1, density filter ft = 2, rmin = 2.4 (reference) | 233.713 | 187.446 | – | – | 0.5000 | 0.5100 | 0.330 | 217 | 2 | 144 | – | – | – |
| MBB top88 60×20 | P0, cell mean, OC as top88 | 237.606 | 187.333 | 187.446 | -0.06 % (J_bin) | 0.5000 | 0.5100 | 0.349 | 219 | 2 | 138 (converged, change_tol) | 139/0/277/– | 2.7 | 0.0191 |
| MBB top88 60×20 | P0, Gauss points, OC as top88 | 235.032 | 188.342 | 187.446 | +0.48 % (J_bin) | 0.5000 | 0.5108 | 0.358 | 220 | 2 | 88 (converged, change_tol) | 89/0/177/– | 1.4 | 0.0162 |
| MBB top88 60×20 | P1, Gauss points, OC as top88 | 237.555 | 187.347 | 187.446 | -0.05 % (J_bin) | 0.5000 | 0.5217 | 0.367 | 219 | 2 | 137 (converged, change_tol) | 138/0/275/– | 1.8 | 0.013 |
| MBB top88 60×20 | P0, cell mean, OC() | 238.978 | 188.369 | 187.446 | +0.49 % (J_bin) | 0.5000 | 0.5108 | 0.360 | 220 | 2 | 72 (converged, objective_range) | 73/0/145/– | 0.8 | 0.0113 |
| MBB top88 60×20 | P0, Gauss points, SiMPL() | 235.069 | 188.056 | 187.446 | +0.33 % (J_bin) | 0.5000 | 0.5125 | 0.357 | 220 | 2 | 39 (converged, objective_range) | 50/0/89/– | 0.4 | 0.00844 |
| MBB top88 60×20 | P1, Gauss points, SiMPL() | 239.291 | 187.242 | 187.446 | -0.11 % (J_bin) | 0.5000 | 0.5242 | 0.377 | 220 | 2 | 36 (converged, objective_range) | 45/0/81/– | 0.4 | 0.00909 |
| MBB top88 150×50 | top88.m in Octave 9.1, density filter ft = 2, rmin = 6.0 (reference) | 235.735 | 189.651 | – | – | 0.5000 | 0.5129 | 0.334 | 549 | 2 | 362 | – | – | – |
| MBB top88 150×50 | P0, cell mean, OC as top88 | 235.363 | 190.528 | 189.651 | +0.46 % (J_bin) | 0.5000 | 0.5077 | 0.326 | 564 | 2 | 457 (converged, change_tol) | 458/0/915/– | 25.0 | 0.0545 |
| MBB top88 150×50 | P0, Gauss points, OC as top88 | 234.75 | 190.473 | 189.651 | +0.43 % (J_bin) | 0.5000 | 0.5076 | 0.326 | 564 | 2 | 543 (converged, change_tol) | 544/0/1087/– | 28.9 | 0.0531 |
| MBB top88 150×50 | P1, Gauss points, OC as top88 | 235.373 | 190.21 | 189.651 | +0.29 % (J_bin) | 0.5000 | 0.5108 | 0.327 | 562 | 2 | 365 (converged, change_tol) | 366/0/731/– | 16.9 | 0.0461 |
| MBB top88 150×50 | P0, cell mean, OC() | 237.021 | 191.381 | 189.651 | +0.91 % (J_bin) | 0.5000 | 0.5093 | 0.332 | 561 | 2 | 58 (converged, objective_range) | 59/0/117/– | 3.2 | 0.0551 |
| MBB top88 150×50 | P0, Gauss points, SiMPL() | 235.542 | 190.702 | 189.651 | +0.55 % (J_bin) | 0.5000 | 0.5087 | 0.329 | 566 | 2 | 45 (converged, objective_range) | 59/0/104/– | 2.6 | 0.044 |
| MBB top88 150×50 | P1, Gauss points, SiMPL() | 235.934 | 191.007 | 189.651 | +0.72 % (J_bin) | 0.5000 | 0.5092 | 0.331 | 564 | 2 | 48 (converged, objective_range) | 59/0/107/– | 2.5 | 0.0419 |

### MBB beam against SiMPL, 768 × 256

Published values (tables 1 and 2 of the paper), compared on J. `validation/mbb_simpl.jl`; Apple M5 Max, 4 threads, Julia 1.13.0, 2026-11-27 20:00.

| case | variant | J | J_bin | ref | vs ref | V | V_bin | grey | perimeter | holes | iterations | solves state/adjoint/filter/step | time (s) | time / state solve (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MBB SiMPL 768×256 | rmin 0.05, P0, SiMPL, authors' sensitivity, S ≤ 1e-5 (table 1) | 0.0012078 | 0.000967674 | 0.0012078 | +0.00 % (J) | 0.3000 | 0.3016 | 0.197 | 17.7 | 6 | 50 (converged, stationarity_tol) | 58/0/108/– | 91.2 | 1.57 |
| MBB SiMPL 768×256 | rmin 0.05, P0, SiMPL, exact sensitivity, S ≤ 1e-5 | 0.00119603 | 0.000964159 | 0.0012078 | -0.97 % (J) | 0.3000 | 0.3005 | 0.190 | 17.2 | 4 | 104 (converged, stationarity_tol) | 114/0/218/– | 174.4 | 1.53 |
| MBB SiMPL 768×256 | rmin 0.05, P1, SiMPL, exact sensitivity, S ≤ 1e-5 | 0.00120303 | 0.000966876 | 0.0012078 | -0.40 % (J) | 0.3000 | 0.3002 | 0.195 | 17.71 | 6 | 73 (converged, stationarity_tol) | 81/0/154/– | 125.3 | 1.55 |
| MBB SiMPL 768×256 | rmin 0.02, P0, SiMPL, authors' sensitivity, 29 iterations (table 2) | 0.00105663 | 0.000971927 | 0.0010566 | +0.00 % (J) | 0.3000 | 0.3008 | 0.089 | 19.03 | 10 | 29 (maxiter) | 31/0/60/– | 51.1 | 1.65 |
| MBB SiMPL 768×256 | rmin 0.05, P0, OC(), default stop (vs OC of table 1) | 0.00121132 | 0.00098081 | 0.0012234 | -0.99 % (J) | 0.3000 | 0.2993 | 0.196 | 17.97 | 7 | 102 (converged, objective_range) | 103/0/205/– | 168.4 | 1.63 |

### 3D cantilever 24 × 12 × 12

No reference; the perimeter is an area and the holes are closed cavities. `validation/cantilever3d.jl`; Apple M5 Max, 4 threads, Julia 1.13.0, 2026-11-27 20:02.

| case | variant | J | J_bin | ref | vs ref | V | V_bin | grey | perimeter | holes | iterations | solves state/adjoint/filter/step | time (s) | time / state solve (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cantilever 3D 24×12×12 | P0, OC() | 1.65812 | 0.874251 | – | – | 0.3000 | 0.2847 | 0.483 | 7.264 | 0 | 82 (converged, objective_range) | 83/0/165/– | 17.8 | 0.215 |
| cantilever 3D 24×12×12 | P0, SiMPL() | 1.27693 | 0.738577 | – | – | 0.3000 | 0.2969 | 0.407 | 5.792 | 0 | 30 (converged, objective_range) | 34/0/64/– | 11.5 | 0.339 |
| cantilever 3D 24×12×12 | P1, OC() | 1.71558 | 0.893944 | – | – | 0.3000 | 0.2784 | 0.500 | 7.264 | 0 | 90 (converged, objective_range) | 91/0/181/– | 27.3 | 0.3 |
| cantilever 3D 24×12×12 | P1, SiMPL() | 1.31487 | 0.777166 | – | – | 0.3000 | 0.2859 | 0.420 | 5.556 | 0 | 30 (converged, objective_range) | 36/0/66/– | 12.7 | 0.353 |

### Bridge of Takezawa et al. (2010), 200 × 100

Published J = 23.3 at 100 iterations, compared on J. `validation/bridge_takezawa.jl`; Apple M5 Max, 4 threads, Julia 1.13.0, 2026-11-27 20:04.

| case | variant | J | J_bin | ref | vs ref | V | V_bin | grey | perimeter | holes | iterations | solves state/adjoint/filter/step | time (s) | time / state solve (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bridge Takezawa 200×100 | RD double well, method default, seed takezawa | 23.6684 | 22.0839 | 23.3 | +1.58 % (J) | 0.3000 | 0.3034 | 0.100 | 18.95 | 9 | 100 (maxiter) | 101/0/0/– | 15.2 | 0.15 |
| bridge Takezawa 200×100 | RD double well, method default, seed hex | 23.8995 | 21.9336 | 23.3 | +2.57 % (J) | 0.3000 | 0.3105 | 0.087 | 16.6 | 6 | 100 (maxiter) | 101/0/0/– | 13.4 | 0.132 |
| bridge Takezawa 200×100 | RD double well, method takezawa2010, seed takezawa | 23.6144 | 21.6804 | 23.3 | +1.35 % (J) | 0.3001 | 0.3072 | 0.104 | 19.73 | 9 | 100 (maxiter) | 101/0/0/– | 16.5 | 0.163 |
| bridge Takezawa 200×100 | RD double well, method takezawa2010, seed hex | 23.6614 | 21.5482 | 23.3 | +1.55 % (J) | 0.3001 | 0.3083 | 0.108 | 20.86 | 8 | 100 (maxiter) | 101/0/0/– | 13.5 | 0.134 |

### Cantilever 2 × 1 with the double well: mesh independence

No reference; J within ±2.7 % of the value on 200 × 100. `validation/mesh_double_well.jl`; Apple M5 Max, 4 threads, Julia 1.13.0, 2026-11-27 20:05.

| case | variant | J | J_bin | ref | vs ref | V | V_bin | grey | perimeter | holes | iterations | solves state/adjoint/filter/step | time (s) | time / state solve (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cantilever 2×1 double well 100×50 | RD double well, κ = 1e-5, T = 25, seed :hex | 2.5582 | 2.43937 | – | – | 0.5000 | 0.5052 | 0.114 | 12.16 | 6 | 92 (converged, all) | 93/0/0/– | 3.4 | 0.0363 |
| cantilever 2×1 double well 200×100 | RD double well, κ = 1e-5, T = 25, seed :hex | 2.49332 | 2.41558 | – | – | 0.5000 | 0.5033 | 0.070 | 13.88 | 8 | 75 (converged, all) | 76/0/0/– | 12.8 | 0.169 |
| cantilever 2×1 double well 400×200 | RD double well, κ = 1e-5, T = 25, seed :hex | 2.42552 | 2.3795 | – | – | 0.5001 | 0.5004 | 0.049 | 16.92 | 14 | 100 (converged, all) | 101/0/0/– | 58.4 | 0.578 |

### Cantilever of levelset88 (Otomori et al. 2015), 80 × 64

levelset88 run in Octave 9.1, another method on the same mesh: rule between methods on `J_bin`. `validation/otomori_levelset88.jl`; Apple M5 Max, 4 threads, Julia 1.13.0, 2026-11-27 20:06.

| case | variant | J | J_bin | ref | vs ref | V | V_bin | grey | perimeter | holes | iterations | solves state/adjoint/filter/step | time (s) | time / state solve (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cantilever levelset88 80×64 | levelset88 in Octave 9.1, stop of Otomori (reference) | 534.949 | 534.949 | – | – | 0.4996 | 0.4996 | 0.000 | 722 | 8 | 112 | – | – | – |
| cantilever levelset88 80×64 | RD V1, Ersatz(), :physical, phi_bc = :zero_boundary | 519.405 | 532.535 | 534.949 | -0.45 % (J_bin) | 0.5001 | 0.4992 | 0.060 | 598 | 5 | 128 (converged, all) | 129/0/0/– | 30.8 | 0.239 |
