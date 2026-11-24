
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
@_export Operators AdjointProblem
@_export Operators SymmetricState
@_export Operators SelfAdjoint
@_export Operators get_adjoint
@_export Operators is_converged
@_export Operators OptContext
@_export Functionals Functional
@_export Functionals Volume
@_export Functionals Constraint
@_export Functionals LinearInState
@_export Optimizers AugmentedLagrangian
@_export Optimizers Bertsekas
@_export Functionals functional_value
@_export Functionals functional_gradient!
@_export Checks check_consistency
@_export PostProcess OptPostProcessor
@_export PostProcess write_step
@_export PostProcess extend_field
@_export PostProcess get_pvd
@_export Optimizers OC
@_export Optimizers SiMPL
@_export Methods Optimization
@_export Methods DensityMethod
@_export Methods ReactionDiffusion
@_export Methods ExactVolume
@_export Methods L1Mean
@_export Methods L2Norm
@_export Methods MaxAbs
@_export Methods LumpedMass
@_export Methods Euclidean
@_export Methods FixedMesh
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
@_export Variables PointwiseMap
@_export Variables Step
@_export Variables Ersatz
@_export Variables design_measures
@_export Operators Parameter
@_export Variables single_field_design
@_export Variables frozen_indicator
@_export Variables tag_adjacent_cells!
@_export Variables get_values
@_export Variables set_values!

@_public Variables get_bounds
@_public Variables DesignMeasures
@_public Variables is_differentiable_field
@_public Variables cell_fraction
@_public Variables changed!
@_public Variables set_beta!
@_public Variables design_gradient!
@_public Variables reset_sink!
@_public Variables add_forms!
@_public Variables finish_sink!
@_public Variables update_fields!
@_public Variables is_linear_field
@_public Operators state_version
@_public Operators invalidate!
@_public Operators refresh!
@_public Methods default_stop
@_public Methods default_metric
@_public Methods monitor_value
@_public Methods metric_weights
@_public Optimizers StepInfo
@_public PostProcess FieldWriter
@_public PostProcess write_fields
@_public Functionals AbstractFunctional
@_public Optimizers MultiplierUpdate
@_public Optimizers end_iteration!
@_public Optimizers objective_changed!
@_public Optimizers handles
@_public Optimizers max_native
@_export Methods OptState
@_public Methods AbstractMethod
@_public Methods stepper
@_public Methods build_oracle
@_public Methods augmented_lagrangian
@_public Methods check_problem
@_public Methods method_label
@_public Methods metric_label
@_public Methods banner_lines
@_public Methods feasible_start
@_public Methods phase_change
@_public Methods material_change
@_public Methods Discretization
@_public Methods RebuildCheck
@_public Methods needs_setup
@_public Methods due
@_public Methods rediscretize
@_public Methods discretization_label
@_public Methods rebuild
