"""
    HyperShape.Methods

Optimisation problems and methods. A method is an operator z ↦ z⁺ over the functionals; all methods share one loop
(evaluate, differentiate, step, accept or reject, continuation, stopping, checkpoint) and only the step is specific:

- [`SIMP`](@ref): density methods; the step is any `Optimizer` of `HyperShape.Optimizers` (OC, SiMPL).

Reaction–diffusion and level-set methods (phase 2) will plug into the same loop.
"""
module Methods

using LinearAlgebra
using TOML
using Printf
using Gridap
using Gridap.FESpaces
import Gridap: solve!
using ..Optimizers: Optimizer, OC, SiMPL, constraint_tol, AbstractOracle, StepInfo, init, step!, reset!, shrink!
import ..Optimizers: objective, gradient!, bounds, linear_constraint, has_constraint, constraint, constraint_gradient!,
                     inner, riesz!
using ..Operators: OptContext, set_context!, StateOperator, num_solves
using ..Variables: OptVariables, Heaviside, set_beta!, get_values, get_bounds, set_values!, get_designfields,
                   get_optvariables, filter_operators
using ..Functionals: Functional, functional_value, functional_gradient!

export Optimization, SIMP, LumpedMass, Euclidean
export Continuation, StopCriteria, RejectStep, Checkpoint, load_checkpoint
export OptHistory, OptResult, OptimizationFailure
export solve!

include("Problem.jl")
include("Loop.jl")
include("Output.jl")

end
