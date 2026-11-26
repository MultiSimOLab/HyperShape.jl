# Mesh independence of the double well (ReactionDiffusion + DoubleWell) with physical parameters: the cantilever 2 × 1
# (cantilever_dw), κ = τ ℓ² = 1e-5 and T = 25 per iteration on 100 × 50, 200 × 100 and 400 × 200, from hole_seed (:hex),
# the default stop (after the settling). The substeps per iteration grow like 1/h² (2, 5, 20); the compliance is the
# same within about 3 %, the topology is not (it depends on the seed and on the mesh). Checks: J in the intervals
# measured when the default was decided (study of 2026-11-25), volume 0.5 ± 1e-3.
#
# julia --project=validation -t 4 validation/mesh_double_well.jl      # about 3 min
include(joinpath(@__DIR__, "cases.jl"))

println("Cantilever 2 × 1 with the double well, κ = 1e-5, T = 25, seed :hex · ", Sys.cpu_info()[1].model, ", ",
        Threads.nthreads(), " threads, Julia ", VERSION, "\n")
println("| mesh | substeps | Δt | J | iterations | settled at | volume | interior holes | time |")
println("|---|---|---|---|---|---|---|---|---|")
# measured (2026-11-26, Mac M5 Max, Julia 1.13, 4 threads): 2.5582, 2.4933, 2.4255 (study of 2026-11-25, which stopped
# at the iteration of the settling: 2.5589, 2.4931, 2.4269)
const MESH_J = Dict(100 => (2.53, 2.59), 200 => (2.47, 2.52), 400 => (2.40, 2.45))
ok = true
Js = Float64[]
for nx in (100, 200, 400)
  c = cantilever_dw(; nx, κ = 1e-5)
  r = c.result
  J = r.objective
  push!(Js, J)
  @printf("| %d × %d | %d | %.4g | %.4f | %d (%s) | %d | %.5f | %d | %.1f s |\n", nx, nx ÷ 2, r.state.nsub, r.state.Δt_nom, J,
          r.iterations, r.status, r.state.settle_iter, c.volume, c.holes, c.time)
  lo, hi = MESH_J[nx]
  global ok &= lo <= J <= hi && abs(c.volume - 0.5) <= 1e-3 && r.status === :converged
end
@printf("\nspread of J: %.2f %% of the mean\n", 100 * (maximum(Js) - minimum(Js)) / (sum(Js) / length(Js)))
println(ok ? "all checks passed" : "SOME CHECK FAILED")
