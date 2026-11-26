# Validation

Long validation cases of HyperShape, outside CI. The fast ones run in the test suite (`test/Validation`). Results and
discussion: the *Validation* page of the documentation.

```
julia --project=validation -e 'using Pkg; Pkg.instantiate()'
julia --project=validation -t 4 validation/mbb_top88.jl      # about 2 min
julia --project=validation -t 4 validation/cantilever3d.jl   # about 1.5 min
julia --project=validation -t 4 validation/mbb_simpl.jl      # about 4 min
julia --project=validation -t 4 validation/bridge_takezawa.jl # about 1.5 min
julia --project=validation -t 4 validation/mesh_double_well.jl # about 3 min
```

| File | Content |
|---|---|
| `cases.jl` | `mbb` (half MBB beam, top88 and SiMPL setups), `cantilever3d`, `bridge` (Takezawa et al. 2010, double well) and `cantilever_dw` (double well); shared with `test/Validation` and `test/ReactionDiffusion` |
| `top88.jl` | plain-Julia port of top88 with the PDE filter, an independent reference |
| `mbb_top88.jl` | MBB 60×20 and 150×50 against top88 with the PDE filter (Andreassen et al. 2011) |
| `mbb_simpl.jl` | MBB 768×256 against SiMPL (Kim, Lazarov, Surowiec, Keith, arXiv:2411.19421) |
| `cantilever3d.jl` | 3D cantilever 24×12×12: Taylor tests, symmetry, OC and SiMPL |
| `bridge_takezawa.jl` | bridge 200×100 with the double well against Takezawa, Nishiwaki and Kitamura (2010): J = 23.3 |
| `mesh_double_well.jl` | cantilever with the double well on 100×50, 200×100 and 400×200 with physical κ and T: mesh independence of J |

Each script prints a Markdown table with the machine, the threads and the Julia version.
