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
julia --project=validation -t 4 validation/otomori_levelset88.jl [outdir] # about 1 min, writes otomori_levelset88.svg
julia --project=validation -t 4 validation/run_all.jl [case ...]   # all of the above, about 13 min
```

Every script also writes its rows of the fixed report table (`<case>.csv` and `<case>.md`) in `validation/results`
(or `ENV["HYPERSHAPE_RESULTS"]`; the folder is not versioned); `run_all.jl` runs each script in its own process and
joins the tables into `results/report.csv` and `results/report.md`. The columns and the rule for comparing methods
are on the *Benchmarks* page of the documentation.

| File | Content |
|---|---|
| `common/common.jl` | includes the shared helpers below (scripts, not a library); also included by `test/Validation` and `test/ReactionDiffusion` |
| `common/problems.jl` | `mbb` (half MBB beam, top88 and SiMPL setups), `top88_reference` (reads `data/top88`), `cantilever3d`, `bridge` (Takezawa et al. 2010, double well) and `cantilever_dw` (double well) |
| `common/measures.jl` | cell means, `binary_evaluator` (`J_bin`: the objective of the design read as 0/1 per cell), `density_evaluator` (the objective of a design given by one density per cell), `binary_volume`, `grey`, `perimeter`, `interior_holes` |
| `common/report.jl` | the fixed report table: `report_row`, `reference_row`, `write_report` (CSV and Markdown) |
| `common/figure.jl` | `design_figure`: SVG of 0/1 designs beside their reference, with histories |
| `run_all.jl` | runs every case and joins the report tables |
| `mbb_top88.jl` | MBB 60×20 and 150×50 against top88 (Andreassen et al. 2011) run in Octave with its density filter (`data/top88`, CSV only: the code is not included) |
| `mbb_simpl.jl` | MBB 768×256 against SiMPL (Kim, Lazarov, Surowiec, Keith, arXiv:2411.19421) |
| `cantilever3d.jl` | 3D cantilever 24×12×12: Taylor tests, symmetry, OC and SiMPL |
| `bridge_takezawa.jl` | bridge 200×100 with the double well against Takezawa, Nishiwaki and Kitamura (2010): J = 23.3 |
| `mesh_double_well.jl` | cantilever with the double well on 100×50, 200×100 and 400×200 with physical κ and T: mesh independence of J |
| `otomori_levelset88.jl` | cantilever 80×64 of levelset88 (Otomori et al. 2015) with `ReactionDiffusion` (`Ersatz()`, `:physical`, `phi_bc = :zero_boundary`) against the literal code run in Octave (`data/levelset88`): compliance, volume and design, table and SVG figure; a comparison, not a reproduction |

Each script prints its Markdown table with the machine, the threads and the Julia version, then the fixed report table.
