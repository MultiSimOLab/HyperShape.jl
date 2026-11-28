# The bridge of Takezawa, Nishiwaki and Kitamura (2010, Section 5.2) with ReactionDiffusion + DoubleWell, 200 × 100,
# 100 iterations, κ = 1e-5 (τ = 5e-6): published J = 23.3. Checks (the tolerance decided for the double
# well, tighter than the 5 % of a published value in the bank: the default reproduces it within 3 %): |J − 23.3|/23.3 ≤ 3 %,
# volume 0.3 ± 1e-4, 5–10 interior holes, the default scheme and the settings of the paper (:takezawa2010, TAKEZAWA2010 of common/presets.jl), from their seed and from :hex.
# The load is the unit point load of the paper (validation/common/problems.jl, `bridge`), kept as it is because it reproduces the
# published value; examples/bridge_dw.jl spreads the load over a fixed width (its J is then independent of the mesh).
# The designs are symmetric about x = 1 up to a few cells (rounding breaks the symmetry and the evolution amplifies
# it): the table gives the fraction of cells whose phase differs from the mirror one.
#
# julia --project=validation -t 4 validation/bridge_takezawa.jl      # exits with code 1 if a check fails
include(joinpath(@__DIR__, "common", "common.jl"))

println("Bridge of Takezawa et al. (2010), 200 × 100, κ = 1e-5, 100 iterations · ", Sys.cpu_info()[1].model, ", ",
        Threads.nthreads(), " threads, Julia ", VERSION, "\n")
println("| method | seed | J | vs 23.3 | volume | interior holes | phase asymmetry | time |")
println("|---|---|---|---|---|---|---|---|")
for method in (:default, :takezawa2010), seed in (:takezawa, :hex)                    # compile (the times exclude it)
  bridge(; nx = 40, seed, method, maxiter = 2)
end
ok = true
rows = []
for method in (:default, :takezawa2010), seed in (:takezawa, :hex)
  b = bridge(; nx = 200, seed, method, maxiter = 100)
  push!(rows, report_row("bridge Takezawa 200×100", "RD double well, method $method, seed $seed", b; ref = 23.3))
  J = b.result.objective
  e = J / 23.3 - 1
  @printf("| %s | %s | %.4f | %+.2f %% | %.5f | %d | %.4f | %.1f s |\n", method, seed, J, 100e, b.volume, b.holes,
          b.asymmetry_phase, b.time)
  global ok &= abs(e) <= 0.03 && abs(b.volume - 0.3) <= 1e-4 && 5 <= b.holes <= 10
end
write_report("bridge_takezawa", rows; title = "Bridge of Takezawa, Nishiwaki and Kitamura (2010), 200 × 100, 100 iterations")
finish(ok, "|J − 23.3| ≤ 3 %, volume 0.3 ± 1e-4, 5–10 interior holes")
