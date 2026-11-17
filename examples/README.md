# Examples

Complete optimisation scripts. Each one defines a function with the size and the output folder as keywords and runs
it when the file is executed:

```
julia --project=examples -t auto examples/mbb.jl
```

- `mbb.jl`: half MBB beam, compliance minimisation with a density method, Helmholtz filter, tanh projection with
  continuation, volume constraint, SiMPL, checkpoints and VTU output (the script of the README).

The test suite runs every example at a small size.
