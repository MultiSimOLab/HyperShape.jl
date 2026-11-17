# MBB beam against SiMPL (Kim, Lazarov, Surowiec, Keith, arXiv:2411.19421v3, table 1: SiMPL-A 1.2078e-3, OC
# 1.2234e-3): 768×256 on (0, 3)×(0, 1), plane strain, volume 0.3, p = 3, Emin = 1e-6, stop at the stationarity measure
# 1e-5. rmin = 0.05, the value used for table 1 (default of MBB2 at dohyun-cse/mfem@28e17f2, the commit with the command
# of table 1); the authors' sensitivity halves the shear term, see the Validation page of the docs.
# Run: julia --project=validation -t 4 validation/mbb_simpl.jl
include(joinpath(@__DIR__, "cases.jl"))
println("Julia $(VERSION), threads $(Threads.nthreads()), $(Sys.cpu_info()[1].model)\n")
mbb(; case = :simpl, nx = 48, ny = 16, vf = 0.3, rmin = 0.1, Emin = 1e-6, maxiter = 2, opt = SiMPL())     # compile
base = (; case = :simpl, nx = 768, ny = 256, vf = 0.3, rmin = 0.05, Emin = 1e-6, maxiter = 300)
println("| Case | C | vs reference | iterations | status | time |\n|---|---|---|---|---|---|")
println(row("768×256 P0, SiMPL (Armijo), kkt 1e-5", mbb(; base..., opt = SiMPL(), stop = StopCriteria(kkt = 1e-5)), 1.2078e-3))
println(row("768×256 P1, SiMPL (Armijo), kkt 1e-5", mbb(; base..., opt = SiMPL(), order = 1, stop = StopCriteria(kkt = 1e-5)), 1.2078e-3))
println(row("768×256 P0, OC(), default stop (vs OC of the paper)", mbb(; base..., opt = OC()), 1.2234e-3))
