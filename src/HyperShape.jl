module HyperShape

using TimerOutputs
using HyperFEM, Gridap

include("TopOpt/TopOpt.jl")
include("Optimizers/Optimizers.jl")
include("Operators/Operators.jl")
include("Variables/Variables.jl")
include("Functionals/Functionals.jl")
include("Checks/Checks.jl")


include("Exports.jl")
end
