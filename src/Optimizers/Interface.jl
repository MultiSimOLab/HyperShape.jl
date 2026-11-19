"""
    Optimizer

Supertype of the optimisers. A subtype implements [`init`](@ref), [`step!`](@ref), [`reset!`](@ref) and
[`shrink!`](@ref).
"""
abstract type Optimizer end

"""
    StepInfo{K}

What [`step!`](@ref) reports about one design step, for an optimiser that handles `K` constraints.

- `accepted`: `false` if no acceptable step was found; `z` is then left unchanged.
- `f`: objective at the design before the step.
- `f_new`: objective at the new design, if the optimiser evaluated it (`NaN` otherwise).
- `kkt`: optimality measure at the design before the step, if the optimiser computes one (`NaN` otherwise).
- `change`: `maximum(abs, z_new - z_old)`.
- `step`: step size (SiMPL) or move limit (OC) used; with adaptive move limits, their mean over all entries.
- `evaluations`: number of calls to `objective` in the step (forward solves).
- `stationarity`: stationarity measure at the design before the step, `S = ‖z − P(z − g)‖_M` (see
  [`stationarity!`](@ref)); `NaN` if the optimiser does not compute it.
- `violations`: for each constraint, its violation at the returned design (see `ConstraintInfo`; `max(0, g(z))` for
  `g ≤ 0`). [`violation`](@ref) is the largest.
- `multipliers`: for each constraint, its multiplier (linearised at the current design if it is nonlinear) used in
  the step: in OC the `λ` of the last bisection (never `0`), in SiMPL the shift `μ` of the volume projection (`0` if
  the constraint is inactive); `NaN` when the step fails.

OC and SiMPL report `StepInfo{1}` (with violation `0` and multiplier `NaN` when the problem has no constraint),
built with `StepInfo(accepted, f, f_new, kkt, change, step, violation, evaluations, multiplier, stationarity)`.
"""
struct StepInfo{K}
  accepted::Bool
  f::Float64
  f_new::Float64
  kkt::Float64
  change::Float64
  step::Float64
  evaluations::Int
  stationarity::Float64
  violations::NTuple{K,Float64}
  multipliers::NTuple{K,Float64}
end

# one constraint (OC, SiMPL)
StepInfo(accepted::Bool, f::Real, f_new::Real, kkt::Real, change::Real, step::Real, violation::Real, evaluations::Integer,
         multiplier::Real, stationarity::Real) =
  StepInfo{1}(accepted, f, f_new, kkt, change, step, evaluations, stationarity, (Float64(violation),), (Float64(multiplier),))

"""
    violation(info::StepInfo) -> Float64

Largest violation of the constraints at the returned design (`0` without constraints).
"""
violation(i::StepInfo) = maximum(i.violations; init = 0.0)


"""
    handles(opt::Optimizer, c::ConstraintInfo) -> Bool

Whether the optimiser treats the constraint `c` itself (natively), as OC and SiMPL do with one inequality `g ≤ 0`
that costs no forward solve (design only). The constraints it does not handle go to the augmented Lagrangian.
Default `false`. `solve!` asks it before scaling the constraints: only the kind of each bound (0, finite or ±Inf),
`linear`, `design_only` and `field` are meaningful then.
"""
handles(::Optimizer, ::ConstraintInfo) = false

"""
    max_native(opt::Optimizer) -> Int

How many constraints the optimiser can treat natively (see [`handles`](@ref)). Default `0`.
"""
max_native(::Optimizer) = 0

"""
    requires_native(opt::Optimizer) -> Bool

Whether the optimiser needs one constraint of its own to work (OC: its bisection is on the multiplier of that
constraint). Default `false`.
"""
requires_native(::Optimizer) = false

"""
    constraint_tol(opt::Optimizer) -> Float64

Tolerance to which the optimiser meets a nonlinear constraint; the loop only declares convergence when the constraint
holds to it. Default `Inf` (no requirement); `OC` and `SiMPL` return their `constraint_tol`.
"""
constraint_tol(::Optimizer) = Inf

"""
    computes_kkt(optimizer) -> Bool

Whether the optimiser reports an optimality measure (`kkt` in [`StepInfo`](@ref)); `StopCriteria(kkt_tol)` needs one.
"""
computes_kkt(::Optimizer) = true

function _check_memory(v, names, n, who)
  keys(v) == names || throw(ArgumentError("$who: the saved memory has entries $(keys(v)), expected $names"))
  for x in values(v)
    x isa Real || length(x) == n || throw(ArgumentError("$who: the saved memory has a vector of $(length(x)) values, z has $n"))
  end
end

# a vector of any type copied into a Vector{Float64} (checkpoints), without scalar indexing
_gather(x) = copyto!(Vector{Float64}(undef, length(x)), x)

"""
    memory(opt::Optimizer, state, z) -> NamedTuple or nothing

Memory of the optimiser after an accepted step to `z`, saved in a checkpoint (and carried across a rebuild of the
problem) so that a resumed run continues as the uninterrupted one: a named tuple whose entries are real numbers or
vectors with one entry per entry of `z` (fields of the design space, which a change of mesh can transfer). Default
`nothing`: not saved, the resumed run starts from `init`.
"""
memory(::Optimizer, state, z) = nothing

"""
    restore_memory!(opt::Optimizer, state, z, mem::NamedTuple) -> state

Restores into `state`, just created by `init(opt, z, oracle)`, the memory `mem` returned by [`memory`](@ref).
"""
restore_memory!(::Optimizer, state, z, mem) = state

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

Called by the loop after it rejected the last step and restored `z` to its value before that step. Rolls back the
memory of the optimiser to the start of that step and multiplies the size of the next step (or the move limit) by
`factor` (`0 < factor < 1`). Consecutive calls accumulate; the reduction is dropped once a step stands (the next
`step!` is called without a `shrink!` in between) or after [`reset!`](@ref).
"""
function shrink! end
