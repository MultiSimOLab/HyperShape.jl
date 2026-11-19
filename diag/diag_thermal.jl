# temporary: trajectory of the thermal augmented Lagrangian test on CI
include(joinpath(@__DIR__, "..", "test", "common.jl"))
using Gridap, HyperFEM, HyperShape
using HyperShape.Optimizers: SiMPL
println("threads ", Threads.nthreads(), " julia ", VERSION, " ", Sys.CPU_NAME)
p = state_problem(; n = 16, init = fill(0.6, 256))
(; θ, dΩ) = p
C0 = functional_value(p.compliance(scale = 1.0), nothing)
B0 = functional_value(p.boundary(scale = 1.0), nothing)
vol = Functional(θ -> ∫(θ.ρ)dΩ, θ; dJdθ = θ -> (δ -> ∫(δ)dΩ), scale = 1.0)
pb = Optimization(objective = vol, constraints = (compliance = Constraint(p.compliance(scale = :bound); max = 1.3C0),
                                                  boundary = Constraint(p.boundary(scale = :bound); max = 1.2B0)))
r = solve!(pb, DensityMethod(SiMPL(); metric = LumpedMass(dΩ)); maxiter = 200, verbose = 0, stop = nothing)
h = r.history
for k in (1:10:191)
  println("it ", lpad(k, 3), "-", k + 9, "  viol_b ", round.(h.violations.boundary[k:k+9]; sigdigits = 3),
          "  viol_c ", round.(h.violations.compliance[k:k+9]; sigdigits = 3))
end
println("cons_b last 10 ", round.(h.constraints.boundary[end-9:end]; sigdigits = 3))
println("mult_b last 10 ", round.(h.multipliers.boundary[end-9:end]; sigdigits = 4), " mult_c ", h.multipliers.compliance[end])
println("obj last 5 ", h.objective[end-4:end], " status ", r.status, " ", r.reason)
