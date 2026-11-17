
# exported names of the submodules
macro _export(mod, name)
  quote
    using HyperShape.$mod: $name
    export $name
  end
end

# public but not exported (HyperShape.name): extension points and functions for advanced scripts
macro _public(mod, name)
  esc(Expr(:block, :(using HyperShape.$mod: $name), Expr(:public, name)))
end

@_export Operators LinearOperator
@_export Operators StateOperator
@_export Operators AdjointOperator
@_export Operators is_converged
@_export Operators OptContext
@_export Functionals Functional
@_export Functionals Volume
@_export Functionals functional_value
@_export Functionals functional_gradient!
@_export Checks check_consistency
@_export PostProcess OptPostProcessor
@_export PostProcess write_step
@_export PostProcess extend_field
@_export PostProcess get_pvd
@_export PostProcess FieldWriter
@_export PostProcess write_fields
@_export Optimizers OC
@_export Optimizers SiMPL
@_export Methods Optimization
@_export Methods DensityMethod
@_export Methods LumpedMass
@_export Methods Euclidean
@_export Methods Continuation
@_export Methods StopCriteria
@_export Methods Grey
@_export Methods RejectStep
@_export Methods Checkpoint
@_export Methods load_checkpoint
@_export Methods OptHistory
@_export Methods OptResult
@_export Methods OptimizationFailure
@_export Methods solve!
@_export Variables FieldVariable
@_export Variables OptVariables
@_export Variables DesignFields
@_export Variables PhysicalFields
@_export Variables Identity
@_export Variables HelmholtzFilter
@_export Variables TanhProjection
@_export Variables CellMean
@_export Variables single_field_design
@_export Variables get_values
@_export Variables set_values!

@_public Variables get_bounds
@_public Variables changed!
@_public Variables set_beta!
@_public Variables design_gradient!
@_public Variables update_fields!
@_public Variables is_linear_field
@_public Operators state_version
@_public Operators invalidate!
@_public Operators refresh!
@_public Methods default_stop
@_public Methods default_metric
@_public Methods monitor_value
