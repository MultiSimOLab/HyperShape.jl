# 3D cantilever 24×12×12 (no published value): Taylor tests, symmetry in y, OC and SiMPL with P0 and P1.
# OC and SiMPL end in different local minima (about 30 % apart): each started from the design of the other stays there.
# Run: julia --project=validation -t 4 validation/cantilever3d.jl
include(joinpath(@__DIR__, "common", "common.jl"))
println("Julia $(VERSION), threads $(Threads.nthreads()), $(Sys.cpu_info()[1].model)\n")
cantilever3d(; n = (4, 2, 2), maxiter = 2)                                                          # compile
for order in (0, 1)
  t = @elapsed rep = cantilever3d(; order, check = true)
  @printf("check_consistency, 24×12×12 P%d: passed = %s (%.1f s)\n", order, rep.passed, t)
end
println("\n| Case | C | — | iterations | status | time | y-asymmetry |\n|---|---|---|---|---|---|---|")
rows = []
for order in (0, 1), opt in (OC(), SiMPL())
  r = cantilever3d(; order, opt)
  println(row("24×12×12 P$order, $(nameof(typeof(opt)))()", r), @sprintf(" %.1e |", r.asymmetry))
  push!(rows, report_row("cantilever 3D 24×12×12", "P$order, $(nameof(typeof(opt)))()", r))
end
write_report("cantilever3d", rows; title = "3D cantilever 24 × 12 × 12 (no reference; perimeter is an area, holes are closed cavities)")
