# Reference of levelset88 (Otomori, Yamada, Izui and Nishiwaki 2015)

Used by `validation/otomori_levelset88.jl`. The two CSV files were produced with Octave 9.1 (2026-11-26) from
`levelset88.m`, the code of Appendix A of M. Otomori, T. Yamada, K. Izui and S. Nishiwaki, *Matlab code for a level
set-based topology optimization method using a reaction diffusion equation*, Struct. Multidisc. Optim. 51:1159–1172
(2015), doi:10.1007/s00158-014-1190-z. The code is distributed with the paper for educational purposes and is not
included here.

Run: `levelset88(80, 64, 0.5, 2e-4)`, that is 80 × 64 elements, Vmax = 0.5, τ = 2e-4, with every other parameter as in
the code (E = 1, Emin = 1e-4, ν = 0.3, nvol = 100, Δt = 0.1, d = −0.02, p = 4, the cantilever clamped on the left side
with unit nodal forces on the 2 round(64/32) + 1 = 5 middle nodes of the right side, φ = 0 on the boundary, the
binary element of the paper: material where the mean of φ over the element is positive). Changes made to run it,
none of which changes the computation:

- the plot of each iteration was turned off (Octave without a display);
- the loop runs its 200 iterations without returning at the stopping test; the test is still evaluated at every
  iteration and the first iteration where it holds is recorded;
- at every iteration the compliance `objective(iterNum)` (= F'U) and the volume fraction `vol` are kept, and the
  binary design of the iteration where the stopping test first holds is kept; both are written with 16 significant
  digits.

Files:

- `ref_ls88_80x64_tau2e-4_hist.csv`: `it, F'U, vol` for the 200 iterations (the compliance and the volume fraction of
  the design before the update of iteration `it`); the header gives the stopping iteration (`stop_it=112`).
- `ref_ls88_80x64_tau2e-4_str_stop.csv`: the binary design (1 material) at `stop_it`, one line, the 64 × 80 array of
  the code in column-major order (row 1 at the top, y = 64).

To regenerate them: obtain `levelset88.m` from the paper, make the three changes above (keep `H(iterNum, :) =
[iterNum, objective(iterNum), vol]` and the design `str(:) > 0` at the first iteration that meets the stopping test),
run it with the arguments above in Octave or MATLAB and write the two files in the format described.
