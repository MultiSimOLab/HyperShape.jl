# Optimisation methods (planned; not included in the package yet).
#
# A method is an operator z ↦ z⁺ over the functionals. Common skeleton: update the variables, evaluate,
# differentiate, step, accept or reject. Only the step is method-specific:
#   SIMP(optimizer)      density
#   ReactionDiffusion    level set updated by a reaction–diffusion equation
#   LevelSet             Hamilton–Jacobi evolution
