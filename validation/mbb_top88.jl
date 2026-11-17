# MBB beam against top88 with the PDE filter (Andreassen et al., SMO 43 (2011) 1–16, section 4.2): volume 0.5, p = 3,
# Emin = 1e-9, rmin = 0.04 nelx, OC with move 0.2, stop at max|Δx| ≤ 0.01. Reference compliances: 237.60 (60×20),
# 235.36 (150×50). Run: julia --project=validation -t 4 validation/mbb_top88.jl
include(joinpath(@__DIR__, "cases.jl"))
println("Julia $(VERSION), threads $(Threads.nthreads()), $(Sys.cpu_info()[1].model)\n")
mbb(; case = :top88, nx = 12, ny = 4, vf = 0.5, rmin = 1.5, Emin = 1e-9, maxiter = 2)        # compile
t88 = (; opt = OC(adapt = nothing), stop = StopCriteria(change = 0.01))
println("| Case | C | vs reference | iterations | status | time |\n|---|---|---|---|---|---|")
for (nx, ny, ref) in ((60, 20, 237.60), (150, 50, 235.36))
  base = (; case = :top88, nx, ny, vf = 0.5, rmin = 0.04nx, Emin = 1e-9)
  tp = @elapsed res = top88pde(nx, ny, 0.5, 0.04nx)
  @printf("| %d×%d top88 port (plain Julia) | %.6g | %+.2f %% | %d | — | %.1f s |\n", nx, ny, compliance(res.p, res.xPhys),
          100 * (compliance(res.p, res.xPhys) / ref - 1), res.loop, tp)
  println(row("$(nx)×$(ny) P0, cell mean, OC as top88", mbb(; base..., t88..., cellmean = true), ref))
  println(row("$(nx)×$(ny) P0, Gauss points, OC as top88", mbb(; base..., t88...), ref))
  println(row("$(nx)×$(ny) P1, Gauss points, OC as top88", mbb(; base..., t88..., order = 1), ref))
  println(row("$(nx)×$(ny) P0, cell mean, OC()", mbb(; base..., cellmean = true), ref))
  println(row("$(nx)×$(ny) P0, Gauss points, SiMPL()", mbb(; base..., opt = SiMPL()), ref))
  println(row("$(nx)×$(ny) P1, Gauss points, SiMPL()", mbb(; base..., opt = SiMPL(), order = 1), ref))
  flush(stdout)
end
