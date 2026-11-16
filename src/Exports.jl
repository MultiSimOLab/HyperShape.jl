
macro publish(mod, name)
  quote
    using HyperShape.$mod: $name
    export $name
  end
end

@publish TopOpt FEFunctional
@publish TopOpt adimensionalize!
@publish TopOpt evaluate_derivative!
@publish TopOpt evaluate_objective
@publish TopOpt evaluate!
@publish TopOpt get_space
@publish TopOpt get_derivative
@publish TopOpt OptimFEVariable
@publish TopOpt update!
@publish TopOpt pushforward!
@publish TopOpt pullback!


@publish Operators LinearOperator
@publish Operators StateOperator
@publish Operators is_converged
@publish Operators OptContext
@publish Functionals Functional
@publish Functionals functional_value
@publish Functionals functional_gradient!
@publish Checks check_consistency
@publish Methods Optimization
@publish Methods SIMP
@publish Methods LumpedMass
@publish Methods Euclidean
@publish Methods Continuation
@publish Methods StopCriteria
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