"""
    Optimizer

Supertype of the optimisers. A subtype implements [`init`](@ref), [`step!`](@ref), [`reset!`](@ref) and
[`shrink!`](@ref).
"""
abstract type Optimizer end

"""
    StepInfo

What [`step!`](@ref) reports about one design step.

- `accepted`: `false` if no acceptable step was found; `z` is then left unchanged.
- `f`: objective at the design before the step.
- `f_new`: objective at the new design, if the optimiser evaluated it (`NaN` otherwise).
- `kkt`: optimality measure, if the optimiser computes one (`NaN` otherwise).
- `change`: `maximum(abs, z_new - z_old)`.
- `step`: step size (SiMPL) or move limit (OC) used.
- `evaluations`: number of objective evaluations in the step (including the one with derivative).
"""
struct StepInfo
  accepted::Bool
  f::Float64
  f_new::Float64
  kkt::Float64
  change::Float64
  step::Float64
  evaluations::Int
end

"""
    init(opt::Optimizer, z, oracle) -> state

Creates the memory of the optimiser for the starting design `z`.
"""
function init end

"""
    step!(opt::Optimizer, state, z, oracle, ctx) -> StepInfo

Overwrites `z` with the next design. `ctx` is the read-only context of the optimisation loop (it may be `nothing`).
"""
function step! end

"""
    reset!(opt::Optimizer, state)

Forgets the history of the optimiser (step-size memory, previous iterates), for example after a continuation jump or
a remeshing. The next step behaves like the first one.
"""
function reset! end

"""
    shrink!(opt::Optimizer, state, factor)

Called after the loop rejected the last step and restored `z` to its value before that step. Rolls back the memory
of the optimiser to the start of the last step and multiplies its step size (or move limit) by `factor` (`0 < factor
< 1`).
"""
function shrink! end

