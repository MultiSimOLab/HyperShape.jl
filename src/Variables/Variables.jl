# Design variables (planned; not included in the package yet).
#
# Three linked levels:
#   OptVariables    z  what the optimiser sees: block vector, scaled, bounded
#   DesignFields    x  design fields on the design mesh (nodal P1 or cell-wise P0)
#   PhysicalFields  θ  physical fields read by the weak forms at the quadrature points
# Maps z → x (scaling, frozen regions, symmetries) and x → θ (filter, projection, interpolation),
# with their transposes for the sensitivities.
