# Reference of top88 (Andreassen et al. 2011)

Used by `validation/mbb_top88.jl` and `test/Validation`. The files were produced with GNU Octave 9.1.0 (2026-11-27)
from `top88.m`, the code of E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov and O. Sigmund, *Efficient
topology optimization in MATLAB using 88 lines of code*, Struct. Multidisc. Optim. 43:1–16 (2011),
doi:10.1007/s00158-010-0594-7 (version of Nov 2010, from www.topopt.dtu.dk). The authors reserve all rights to the
code; it is not included here.

Runs: `top88(60, 20, 0.5, 3, 2.4, 2)` and `top88(150, 50, 0.5, 3, 6.0, 2)`: the half MBB beam of the code (unit square
elements, E0 = 1, Emin = 1e-9, ν = 0.3, plane stress, unit point load at the top-left node, symmetry on the left side,
roller at the bottom-right node), volume fraction 0.5, penalisation 3, rmin = 0.04 nelx, the **density filter of the
code (ft = 2, a convolution filter with weights rmin − distance)**, OC with move 0.2, stop at max|Δx| ≤ 0.01. Note:
the validation of HyperShape uses a Helmholtz (PDE) filter, the variant of section 4.2 of the paper, whose code is not
part of `top88.m`; the compliances of the two differ by up to about 2 % (see the Validation page).

The code was run unmodified. To run it without a display and keep its numbers, the functions it calls for printing
and plotting (`fprintf`, `colormap`, `imagesc`, `caxis`, `axis`, `drawnow`) were shadowed by stand-ins in the folder
of the driver: `fprintf` keeps the four numbers of each iteration at full precision and `imagesc` keeps the image
`1 − xPhys`. Nothing of the computation is changed.

Files (17 significant digits):

- `top88_<nelx>x<nely>_hist.csv`: `it, c, vol, change` as the code prints them at each iteration: the compliance of
  the physical design before the update of iteration `it`, the volume fraction and the change after it.
- `top88_<nelx>x<nely>_xPhys.csv`: the final physical design `xPhys`, one value per line, the `nely × nelx` array of
  the code in column-major order (row 1 at the top).

To regenerate them: obtain `top88.m`, put next to the driver the stand-ins described above, call
`top88(nelx, nely, 0.5, 3, rmin, 2)` with the arguments above in Octave or MATLAB and write the two files in the format
described.
