# Optimisers (planned; not included in the package yet).
#
# Independent of finite elements: an optimiser sees vectors, bounds and an oracle.
#   abstract type Optimizer end
#   init(opt, z, oracle) -> state
#   step!(opt, state, z, oracle, ctx) -> status
#   reset!(opt, state)
#   shrink!(opt, state, factor)
# First: optimality criteria (OC) and SiMPL. Later: GCMMA, augmented Lagrangian.
