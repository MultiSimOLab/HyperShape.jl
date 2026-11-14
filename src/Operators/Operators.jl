# Operators over HyperFEM computational models (planned; not included in the package yet).
#
#   StateOperator   forward and adjoint states: the model carries its full weak forms;
#                   solve = (model, ctx) -> ...  supplied by the user
#   LinearOperator  auxiliary linear solves (filter, reaction–diffusion step, velocity extension):
#                   the model carries the bilinear form only and HyperShape supplies the right-hand side;
#                   solve = (model, b, ctx, stale) -> ...  supplied by the user
# Both declare their dependencies explicitly (depends = ...).
