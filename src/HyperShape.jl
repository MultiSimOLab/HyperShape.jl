module HyperShape

using HyperFEM, Gridap

include("Optimizers/Optimizers.jl")
include("Operators/Operators.jl")
include("Variables/Variables.jl")
include("Functionals/Functionals.jl")
include("Checks/Checks.jl")
include("PostProcess/PostProcess.jl")
include("Methods/Methods.jl")


include("Exports.jl")
end
