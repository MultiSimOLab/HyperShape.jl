# Examples

Complete optimisation scripts. Each one defines a function with the size and the output folder as keywords and runs
it when the file is executed:

```
julia --project=examples -e 'using Pkg; Pkg.instantiate()'     # once
julia --project=examples -t auto examples/mbb.jl
```

- `mbb.jl`: half MBB beam, compliance minimisation with a density method, Helmholtz filter, tanh projection with
  continuation, volume constraint, SiMPL, checkpoints and VTU output (the script of the README).
- `cantilever_rd.jl`: cantilever, compliance minimisation with the reaction–diffusion method: a level set through the
  ersatz material (`Ersatz()`), solid start, frozen cells under the load, exact volume lowered 1 % per iteration.

The test suite runs every example at a small size.
