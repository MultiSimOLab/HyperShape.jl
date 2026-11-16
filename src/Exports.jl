
macro publish(mod, name)
  quote
    using HyperShape.$mod: $name
    export $name
  end
end

@publish Operators LinearOperator
@publish Operators StateOperator
@publish Operators AdjointOperator
@publish Operators is_converged
@publish Operators OptContext
@publish Functionals Functional
@publish Functionals Volume
@publish Functionals functional_value
@publish Functionals functional_gradient!
@publish Checks check_consistency
@publish PostProcess OptPostProcessor
@publish PostProcess write_step
@publish PostProcess extend_field
@publish PostProcess get_pvd
@publish PostProcess FieldWriter
@publish PostProcess write_fields
@publish Optimizers OC
@publish Optimizers SiMPL
@publish Methods Optimization
@publish Methods SIMP
@publish Methods LumpedMass
@publish Methods Euclidean
@publish Methods Continuation
@publish Methods StopCriteria
@publish Methods Grey
@publish Methods RejectStep
@publish Methods Checkpoint
@publish Methods load_checkpoint
@publish Methods OptHistory
@publish Methods OptResult
@publish Methods OptimizationFailure
@publish Methods solve!
@publish Variables FieldVariable
@publish Variables OptVariables
@publish Variables DesignFields
@publish Variables PhysicalFields
@publish Variables Identity
@publish Variables HelmholtzFilter
@publish Variables Heaviside
@publish Variables single_field_design

# @publish LinearSolvers solve
# @publish LinearSolvers solve!