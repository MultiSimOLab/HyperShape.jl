"""
    HyperShape.Methods

Optimisation problems and methods. A method is an operator z ↦ z⁺ over the functionals; all methods share one loop
(evaluate, differentiate, step, accept or reject, continuation, stopping, checkpoint) and only the step is specific:

- [`DensityMethod`](@ref): density methods; the step is any `Optimizer` of `HyperShape.Optimizers` (OC, SiMPL).
- [`ReactionDiffusion`](@ref): reaction–diffusion of a level set (through an interface map) or of a density, with
  the volume met exactly at every step.

Level-set methods (phase 2) will plug into the same loop.
"""
module Methods

using LinearAlgebra
using TOML
using Printf
using Gridap
using Gridap.FESpaces
import Gridap: solve!
using ..Optimizers: Optimizer, OC, SiMPL, constraint_tol, computes_kkt, set_move!, _valid_move, memory, restore_memory!, AbstractOracle, StepInfo, init, step!, reset!, shrink!,
                   ConstraintInfo, violation, handles, max_native, AugmentedLagrangian, AugmentedLagrangianOracle,
                   Bertsekas, end_iteration!, objective_changed!, requires_native, is_upper
import ..Optimizers
import ..Optimizers: objective, gradient!, bounds, num_constraints, constraint_info, linear_constraint, constraint,
                     constraint_gradient!, weighted_gradient!, inner, riesz!
using ..Operators: OptContext, set_context!, StateOperator, num_solves, Parameter
using Gridap.Algebra: LinearSolver, symbolic_setup, numerical_setup, numerical_setup!
using ..Variables: PhysicalFields, OptVariables, DesignFields, DesignMeasures, TanhProjection, TanhStage, is_differentiable,
                   has_ersatz, update_fields!, set_beta!, get_values, get_bounds, set_values!, get_designfields,
                   get_optvariables, filter_operators, IdentityStage, StepStage, ErsatzStage, MaskStage, _fraction,
                   _cell_values
using ..Functionals: AbstractFunctional, Functional, Constraint, functional_value, functional_gradient!, get_scale, is_volume,
                     is_design_only, is_linear, physical_fields, responses, _reference
using ..PostProcess: OptPostProcessor, _post_start!, _post_due, _post_call!, _post_close!, _SetupDriver
using Gridap.Geometry: get_background_model
import ..Checks: check_consistency

export Optimization, DensityMethod, LumpedMass, Euclidean, FixedMesh
export Continuation, StopCriteria, RejectStep, Checkpoint, load_checkpoint, Grey
export OptHistory, OptResult, OptimizationFailure, OptState
export solve!
export ReactionDiffusion, ExactVolume, L1Mean, L2Norm, MaxAbs

include("Problem.jl")
include("Discretization.jl")
include("Loop.jl")
include("Output.jl")
include("ReactionDiffusion.jl")

end
