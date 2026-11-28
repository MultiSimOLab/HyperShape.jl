# Shared helpers of the validation cases and benchmarks: included as scripts by validation/*.jl, run_all.jl and
# test/Validation; not a library and not part of HyperShape.
#
#   problems.jl   the problems (geometry, loads, physics, design and method of each case)
#   measures.jl   measures of a design: cell means, J_bin, binary volume, grey, perimeter, interior holes
#   report.jl     the fixed report table (CSV and Markdown)
#   figure.jl     SVG figures of 0/1 designs beside their reference, with histories
using Gridap, Gridap.Geometry, HyperFEM, HyperShape, Printf

include(joinpath(@__DIR__, "measures.jl"))
include(joinpath(@__DIR__, "report.jl"))
include(joinpath(@__DIR__, "figure.jl"))
include(joinpath(@__DIR__, "problems.jl"))
