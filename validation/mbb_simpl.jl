# MBB beam against SiMPL (Kim, Lazarov, Surowiec, Keith, arXiv:2411.19421v3; table 1: SiMPL 1.2078e-3 in 50 iterations,
# OC 1.2234e-3 after 300; table 2, h = 1/256, 30 iterations counting the initial design, 29 steps: 1.0566e-3): 768×256 on (0, 3)×(0, 1), plane strain,
# volume 0.3, p = 3, Emin = 1e-6, P0 control, stop at the stationarity measure Sₖ ≤ 1e-5 (L2 metric). Table 1 used
# rmin = 0.05 (dohyun-cse/mfem 28e17f2, the commit with its command) and the sensitivity of the authors' code, which
# halves the shear term; table 2 used rmin = 0.02. See the Validation page of the docs.
# Run: julia --project=validation -t 4 validation/mbb_simpl.jl
include(joinpath(@__DIR__, "common", "common.jl"))
println("Julia $(VERSION), threads $(Threads.nthreads()), $(Sys.cpu_info()[1].model)\n")
mbb(; case = :simpl, nx = 48, ny = 16, vf = 0.3, rmin = 0.1, Emin = 1e-6, maxiter = 2, opt = SiMPL())     # compile
base = (; case = :simpl, nx = 768, ny = 256, vf = 0.3, Emin = 1e-6, maxiter = 300)
S5 = StopCriteria(stationarity_tol = 1e-5)
println("| Case | C | vs reference | iterations | status | time |\n|---|---|---|---|---|---|")
rows = []
for (name, kw, ref) in (("rmin 0.05, P0, SiMPL, authors' sensitivity, S ≤ 1e-5 (table 1)", (; rmin = 0.05, opt = SiMPL(), sensitivity = :authors, stop = S5), 1.2078e-3),
                        ("rmin 0.05, P0, SiMPL, exact sensitivity, S ≤ 1e-5", (; rmin = 0.05, opt = SiMPL(), stop = S5), 1.2078e-3),
                        ("rmin 0.05, P1, SiMPL, exact sensitivity, S ≤ 1e-5", (; rmin = 0.05, order = 1, opt = SiMPL(), stop = S5), 1.2078e-3),
                        ("rmin 0.02, P0, SiMPL, authors' sensitivity, 29 iterations (table 2)", (; rmin = 0.02, opt = SiMPL(), sensitivity = :authors, maxiter = 29, stop = nothing), 1.0566e-3),
                        ("rmin 0.05, P0, OC(), default stop (vs OC of table 1)", (; rmin = 0.05, opt = OC()), 1.2234e-3))
  c = mbb(; base..., kw...)
  println(row(name, c, ref))
  push!(rows, report_row("MBB SiMPL 768×256", name, c; ref))
end
write_report("mbb_simpl", rows; title = "MBB beam against SiMPL (Kim, Lazarov, Surowiec, Keith)")
