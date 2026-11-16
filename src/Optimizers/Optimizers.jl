"""
    HyperShape.Optimizers

Optimisers for problems with a box-bounded design vector `z` and at most one constraint `g(z) ≤ 0`, linear or not
(for example the volume of a projected density).

The module is independent of finite elements: an optimiser sees vectors, bounds and an oracle (see
[`AbstractOracle`](@ref)). A new optimiser is a subtype of [`Optimizer`](@ref) that implements the four functions of
the contract:

```julia
state = init(opt, z, oracle)            # memory and history
info  = step!(opt, state, z, oracle, ctx) # overwrites z with the next design, returns a StepInfo
reset!(opt, state)                      # forget the history (continuation jump, remeshing)
shrink!(opt, state, factor)             # roll back the last step and reduce the next one
```
"""
module Optimizers

using LinearAlgebra

export Optimizer, StepInfo
export AbstractOracle, FunctionOracle
export objective, gradient!, bounds, linear_constraint, has_constraint, constraint, constraint_gradient!, riesz!
export OC, SiMPL, constraint_tol, set_move!

include("Oracle.jl")
include("Interface.jl")
include("OC.jl")
include("SiMPL.jl")

end
