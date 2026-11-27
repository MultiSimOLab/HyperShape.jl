# 3D cantilever 24×12×12 (no published value): Taylor tests, symmetry in y, OC and SiMPL with P0 and P1.
# OC and SiMPL end in different local minima (about 30 % apart): each started from the design of the other stays there.
# No reference value: the checks are those of the case (consistency of the derivatives, symmetry in y to 1e-6, every
# run converged with the volume at 0.3 ± 1e-3).
# Run: julia --project=validation -t 4 validation/cantilever3d.jl     # exits with code 1 if a check fails
include(joinpath(@__DIR__, "common", "common.jl"))
println("Julia $(VERSION), threads $(Threads.nthreads()), $(Sys.cpu_info()[1].model)\n")
for order in (0, 1), opt in (OC(), SiMPL())                                     # compile every variant (the times exclude it)
  cantilever3d(; n = (4, 2, 2), order, opt, maxiter = 2)
end
ok = true
for order in (0, 1)
  t = @elapsed rep = cantilever3d(; order, check = true)
  @printf("check_consistency, 24×12×12 P%d: passed = %s (%.1f s)\n", order, rep.passed, t)
  global ok &= rep.passed
end
println("\n| Case | C | — | iterations | status | time | y-asymmetry |\n|---|---|---|---|---|---|---|")
rows = []
for order in (0, 1), opt in (OC(), SiMPL())
  r = cantilever3d(; order, opt)
  println(row("24×12×12 P$order, $(nameof(typeof(opt)))()", r), @sprintf(" %.1e |", r.asymmetry))
  q = report_row("cantilever 3D 24×12×12", "P$order, $(nameof(typeof(opt)))()", r)
  push!(rows, q)
  global ok &= r.result.status === :converged && r.asymmetry <= 1e-6 && abs(q.V - 0.3) <= 1e-3
end
write_report("cantilever3d", rows; title = "3D cantilever 24 × 12 × 12 (no reference; perimeter is an area, holes are closed cavities)")
finish(ok, "consistency checks, converged, symmetry in y to 1e-6, volume 0.3 ± 1e-3")
