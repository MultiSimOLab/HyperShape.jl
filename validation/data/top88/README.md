# Reference of top88 (Andreassen et al. 2011)

Used by `validation/mbb_top88.jl` and `test/Validation`. The files were produced with GNU Octave 9.1.0 (2026-11-27)
from `top88.m`, the code of E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov and O. Sigmund, *Efficient
topology optimization in MATLAB using 88 lines of code*, Struct. Multidisc. Optim. 43:1–16 (2011),
doi:10.1007/s00158-010-0594-7 (version of Nov 2010, from www.topopt.dtu.dk). The authors reserve all rights to the
code; it is not included here.

Runs: `top88(60, 20, 0.5, 3, 2.4, 2)` and `top88(150, 50, 0.5, 3, 6.0, 2)`: the half MBB beam of the code (unit square
elements, E0 = 1, Emin = 1e-9, ν = 0.3, plane stress, unit point load at the top-left node, symmetry on the left side,
roller at the bottom-right node), volume fraction 0.5, penalisation 3, rmin = 0.04 nelx, the **density filter of the
code (ft = 2, a convolution filter with weights rmin − distance)**, OC with move 0.2, stop at max|Δx| ≤ 0.01.

**Check against the paper.** The last compliance the code prints, 233.7146 (60×20, 144 iterations) and 235.7337
(150×50, 362 iterations), is the value of Fig. 3 of the paper (density filter): 233.71 and 235.73. (The compliances
of Fig. 4, 237.60 and 235.36, are those of the PDE filter of section 4.2, which is not part of `top88.m`; HyperShape
is compared with them directly, see the Validation page.)

The code was run unmodified. To run it without a display and keep its numbers, the functions it calls for printing
and plotting (`fprintf`, `colormap`, `imagesc`, `caxis`, `axis`, `drawnow`) are shadowed by the stand-ins of the folder
`octave/` (code of HyperShape, not derived from top88): `fprintf` keeps the four numbers of each iteration and
`imagesc` keeps the image `1 − xPhys`. Nothing of the computation is changed.

Files:

- `top88_<nelx>x<nely>_hist.csv`: `it, c, vol, change` at each iteration, written with 17 significant digits: the
  compliance of the physical design before the update of iteration `it`, the volume fraction and the change after it.
- `top88_<nelx>x<nely>_xPhys.csv`: the final physical design `xPhys`, one value per line, the `nely × nelx` array of
  the code in column-major order (row 1 at the top). It is recovered as `1 − (1 − xPhys)` from the image, so its
  values are exact to an absolute 1e-16 (not to 17 relative digits: values near 0 lose their small digits).

To regenerate them with your own copy of `top88.m` (in Octave or MATLAB), from the folder `octave/`:

```
run_top88('/path/to/folder/of/top88', 60, 20, 0.5, 3, 2.4, 2, '60x20')
run_top88('/path/to/folder/of/top88', 150, 50, 0.5, 3, 6.0, 2, '150x50')
```

which writes the four files of this folder.
