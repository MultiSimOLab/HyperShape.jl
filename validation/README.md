# Validation

Long validation cases of HyperShape, outside CI. The fast ones run in the test suite (`test/Validation`). Results and
discussion: the *Validation* page of the documentation.

```
julia --project=validation -e 'using Pkg; Pkg.instantiate()'
julia --project=validation -t 4 validation/mbb_top88.jl      # about 2 min
julia --project=validation -t 4 validation/cantilever3d.jl   # about 1.5 min
julia --project=validation -t 4 validation/mbb_simpl.jl      # about 4 min
```

| File | Content |
|---|---|
| `cases.jl` | `mbb` (half MBB beam, top88 and SiMPL setups) and `cantilever3d`; shared with `test/Validation` |
| `top88.jl` | plain-Julia port of top88 with the PDE filter, an independent reference |
| `mbb_top88.jl` | MBB 60×20 and 150×50 against top88 with the PDE filter (Andreassen et al. 2011) |
| `mbb_simpl.jl` | MBB 768×256 against SiMPL (Kim, Lazarov, Surowiec, Keith, arXiv:2411.19421) |
| `cantilever3d.jl` | 3D cantilever 24×12×12: Taylor tests, symmetry, OC and SiMPL |

Each script prints a Markdown table with the machine, the threads and the Julia version.
