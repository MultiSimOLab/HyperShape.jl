# The common optimisation loop: continuation, step rejection, stopping, history, checkpoint.

"""
    RejectStep(; shrink = 0.5, max_attempts = 5)

What to do when the forward problem does not converge at the design proposed by the optimiser: go back to the
previous design, call `shrink!` on the optimiser with `shrink` (`0 < shrink < 1`: OC multiplies its move limit) and
try again, at most `max_attempts` attempts per iteration; then throw [`OptimizationFailure`](@ref). `RejectStep()` is the
default of [`solve!`](@ref) (with the defaults, the step can go down to 1/16 of the first one).

Optimisers that evaluate their trials (SiMPL) handle failures inside their line search, so for them this only
applies if a step is accepted at a design that cannot be evaluated again.
"""
struct RejectStep
  shrink::Float64
  max_attempts::Int
  function RejectStep(shrink::Real, max_attempts::Integer)
    0 < shrink < 1 || throw(ArgumentError("RejectStep: shrink must be in (0, 1)"))
    max_attempts >= 1 || throw(ArgumentError("RejectStep: max_attempts must be at least 1"))
    new(Float64(shrink), Int(max_attempts))
  end
end

RejectStep(; shrink = 0.5, max_attempts = 5) = RejectStep(shrink, max_attempts)

"""
    StopCriteria(; objective_range = nothing, window = 10, kkt_tol = nothing, kkt_rtol = nothing,
                 stationarity_tol = nothing, stationarity_rtol = nothing, change_tol = nothing)

Convergence criteria; each one is off unless given, and the first one met stops the optimisation (the reason in
[`OptResult`](@ref) is the name of the keyword). With a [`Continuation`](@ref) they only act in the last stage, unless
it was built with `close_early = true` (then a criterion met in an earlier stage ends that stage).

- `objective_range`: range of the objective over the last `window` iterations of the current stage, `(max - min)`,
  relative to `max(largest |f| in the window, 1e-3 |f₁|)` (`f₁` the objective of the first iteration of the run, so an
  objective that tends to zero still converges), below this value. A range, not the difference between the ends of
  the window, so an oscillation does not pass for convergence.
- `kkt_tol`: optimality measure of the optimiser below this value (absolute: units of the scaled objective).
  `kkt_rtol`: the same, relative to the one of the first iteration of the run (off if that one is not a positive
  number). Only with optimisers that compute it (SiMPL; with OC `solve!` throws before the first iteration).
- `stationarity_tol`: the stationarity measure of the paper of SiMPL, `S = ‖z − P(z − g)‖_M`
  ([`stationarity!`](@ref HyperShape.Optimizers.stationarity!), computed by OC and SiMPL at the design before the step,
  so the run stops one iteration after it is met), below this value (absolute: it depends on `scale` and on the
  metric; with `Euclidean()` it depends on the mesh, use `LumpedMass(dΩ)`). `stationarity_rtol`: relative to the first
  finite value of the run. With a nonlinear constraint the projection uses the constraint linearised at `z`. OC
  computes it with the metric of [`DensityMethod`](@ref) (its step does not use it): give `LumpedMass(dΩ)` for a value
  independent of the mesh, comparable with SiMPL.
- `change_tol`: `maximum(abs, Δz)` of the step below this value. With a tanh projection it is rarely reached.

The tolerances are not checked at the first iteration of a stage (after a continuation jump the first step restores
feasibility and its measures say little); `objective_range` needs `window` iterations of the stage.
"""
struct StopCriteria
  objective_range::Union{Nothing,Float64}
  window::Int
  kkt_tol::Union{Nothing,Float64}
  kkt_rtol::Union{Nothing,Float64}
  stationarity_tol::Union{Nothing,Float64}
  stationarity_rtol::Union{Nothing,Float64}
  change_tol::Union{Nothing,Float64}
  function StopCriteria(objective_range, window, kkt_tol, kkt_rtol, stationarity_tol, stationarity_rtol, change_tol)
    for (k, v) in ((:objective_range, objective_range), (:kkt_tol, kkt_tol), (:kkt_rtol, kkt_rtol),
                   (:stationarity_tol, stationarity_tol), (:stationarity_rtol, stationarity_rtol), (:change_tol, change_tol))
      v === nothing || v > 0 || throw(ArgumentError("StopCriteria: $k must be positive"))
    end
    window >= 2 || throw(ArgumentError("StopCriteria: window must be at least 2"))
    f(v) = v === nothing ? nothing : Float64(v)
    new(f(objective_range), Int(window), f(kkt_tol), f(kkt_rtol), f(stationarity_tol), f(stationarity_rtol), f(change_tol))
  end
end

StopCriteria(; objective_range = nothing, window = 10, kkt_tol = nothing, kkt_rtol = nothing, stationarity_tol = nothing,
             stationarity_rtol = nothing, change_tol = nothing) =
  StopCriteria(objective_range, window, kkt_tol, kkt_rtol, stationarity_tol, stationarity_rtol, change_tol)

"""
    default_stop(optimizer) -> Union{StopCriteria,Nothing}

Stopping criteria of [`solve!`](@ref) when `stop` is not given: `StopCriteria(objective_range = 1e-3)` (range of
the objective below 0.1 % over the last 10 iterations of the last stage), for every optimiser. With a projection,
`kkt_tol` and `change_tol` are rarely reached; the objective settles.
"""
default_stop(::Optimizer) = StopCriteria(objective_range = 1e-3)

# `stop` and `on_failure` not given (`stop = nothing` means maxiter only; `on_failure` must be a RejectStep)
struct _DefaultStop end
struct _DefaultFailure end

const HISTORY_COLUMNS = (:iteration, :objective, :kkt, :stationarity, :change, :step, :attempts, :forward, :adjoint,
                         :filter)

"""
    Continuation(; every, close_early = false, name = object => values, …)

Continuation of parameters in stages: `Continuation(β = proj => [1, 2, 4, 8, 16, 32]; every = 40)` (with `proj` a `TanhProjection`) uses `β = 1` in the
first stage, `2` in the second, and so on. A stage ends after `every` iterations, or when the optimiser finds no
acceptable step; with `close_early = true`, also when the [`StopCriteria`](@ref) are met (by default they only act in
the last stage: early stages can look converged on a grey plateau before the topology forms). The optimisation can
only stop in the last stage. At each jump the memory of the
optimiser is forgotten (`reset!`), which costs one more forward solve. All value lists have the same length (the
number of stages). Supported objects: [`Parameter`](@ref) (any number of the user: the penalty of the interpolation,
a load factor), [`TanhProjection`](@ref) (β, through `set_beta!`) and `OC` (its move limit, through
`set_move!`: `Continuation(β = proj => [1, 2, 4, 8, 16], move = oc => [0.2, 0.2, 0.1, 0.05, 0.025]; every = 50)`, the
same `oc` given to `DensityMethod`).
"""
struct Continuation{N,O<:Tuple,V<:Tuple}
  names::NTuple{N,Symbol}
  objects::O
  values::V
  every::Int
  nstages::Int
  close_early::Bool
end

function Continuation(; every::Integer, close_early::Bool = false, kwargs...)
  isempty(kwargs) && throw(ArgumentError("Continuation: give at least one parameter, for example β = proj => [1, 2, 4]"))
  every >= 1 || throw(ArgumentError("Continuation: every must be at least 1"))
  names = Tuple(keys(kwargs))
  pairs_ = Tuple(values(kwargs))
  all(p -> p isa Pair, pairs_) || throw(ArgumentError("Continuation: each parameter is object => values"))
  objects = Tuple(p.first for p in pairs_)
  vals = Tuple(Float64.(collect(p.second)) for p in pairs_)
  for o in objects
    applicable(_set_parameter!, o, 1.0) || throw(ArgumentError("Continuation: cannot continue a parameter of $(typeof(o))"))
  end
  for n in names
    n in HISTORY_COLUMNS && throw(ArgumentError("Continuation: $n is the name of a history column; choose another name"))
  end
  allunique(objectid.(objects)) || throw(ArgumentError("Continuation: the same object is continued under two names"))
  for (o, v) in zip(objects, vals), x in v
    _valid_parameter(o, x) || throw(ArgumentError("Continuation: value $x not valid for $(typeof(o))"))
  end
  n = length(first(vals))
  n >= 1 && all(v -> length(v) == n, vals) ||
    throw(ArgumentError("Continuation: every parameter needs the same number of values (stages)"))
  Continuation(names, objects, vals, Int(every), n, close_early)
end

_set_parameter!(h::TanhProjection, v) = set_beta!(h, v)
_valid_parameter(h::TanhProjection, v) = isfinite(v) && v > 0
_set_parameter!(o::OC, v) = set_move!(o, v)
_set_parameter!(p::Parameter, v) = (p[] = v; p)
_valid_parameter(p::Parameter, v) = isfinite(v)
_valid_parameter(o::OC, v) = _valid_move(o, v)

Base.show(io::IO, c::Continuation) =
  print(io, "Continuation(", join(c.names, ", "), "; ", c.nstages, " stages, every ", c.every, ")")

function _apply_stage!(c::Continuation, s, buf)
  foreach((o, v) -> _set_parameter!(o, v[s]), c.objects, c.values)
  for (j, v) in enumerate(c.values)
    buf[j] = v[s]
  end
  NamedTuple{c.names}(Tuple(v[s] for v in c.values))
end

"""
    Checkpoint(; filepath, every)

Writes the state of the optimisation to the folder `filepath` every `every` accepted iterations, and also when the
optimisation stops or fails: `checkpoint.toml` (format version 2: iteration, continuation stage, scales of the
functionals by name, who treats each constraint and the multipliers and penalties of the augmented Lagrangian by
name, sizes and the names of the other files), `z_<iteration>.bin` (the design, raw `Float64`), `history_<iteration>.csv`
and, for optimisers that save one, `memory_<iteration>.bin` (see [`memory`](@ref HyperShape.Optimizers.memory)).
The data files are written first and `checkpoint.toml` last, under a temporary name and then renamed, so they always
belong to the same iteration; older files are then deleted. Resume with `solve!(…; resume = filepath)` (or the
result of [`load_checkpoint`](@ref)).
"""
struct Checkpoint
  filepath::String
  every::Int
  function Checkpoint(filepath::AbstractString, every::Integer)
    every >= 1 || throw(ArgumentError("Checkpoint: every must be at least 1"))
    new(String(filepath), Int(every))
  end
end

Checkpoint(; filepath::AbstractString, every::Integer) = Checkpoint(filepath, every)
Base.show(io::IO, c::Checkpoint) = print(io, "Checkpoint(filepath = ", repr(c.filepath), ", every = ", c.every, ")")

"""
    Grey(θ::PhysicalFields, dΩ; field = nothing)

Monitor of the measure of grey of a physical density, `M_nd = 4/|Ω| ∫ ρ (1 - ρ) dΩ` (Sigmund 2007): 0 for a 0/1
design, 1 for a uniform 0.5. Pass it to `solve!(…; monitors = (grey = Grey(θ, dΩ),))` to add it to the history, the
table and the closing line; it is not a stopping criterion. `field` is mandatory when θ has several fields. Costs one integral per
iteration (about 90 kB of memory, independent of the mesh).
"""
struct Grey{T,M}
  θ::T
  field::Symbol
  measure::M
  volume::Float64
end

function Grey(θ::PhysicalFields, dΩ; field = nothing)
  names = propertynames(θ)
  f = field === nothing ?
      (length(names) == 1 ? names[1] : throw(ArgumentError("Grey: several fields ($(join(names, ", "))); give field = …"))) :
      field
  f in names || throw(ArgumentError("Grey: no field named $f"))
  Grey(θ, f, dΩ, Float64(sum(∫(1.0)dΩ)))
end

grey_value(g::Grey) = (ρ = getproperty(g.θ, g.field); 4 * sum(∫(ρ * (1 - ρ))g.measure) / g.volume)

"""
    monitor_value(m) -> Real

Value of a monitor of [`solve!`](@ref) (`monitors = (name = m, …)`) at the accepted design, called once per accepted
iteration after the forward solve (the closing line shows the last value recorded). It must return a real number; an
exception ends the optimisation as an [`OptimizationFailure`](@ref) with the history and the checkpoint written.
Defined for [`Grey`](@ref).
"""
monitor_value(g::Grey) = grey_value(g)

Base.show(io::IO, g::Grey) = print(io, "Grey(", g.field, ")")

function _monitor(n, m)
  v = monitor_value(m)
  v isa Real || throw(ArgumentError("monitor $n returned a $(typeof(v)), not a number"))
  Float64(v)
end

_last_monitors(hist, monitors) =
  NamedTuple{keys(monitors)}(Tuple(hist.n == 0 ? NaN : hist.monitor_values[j, hist.n] for j in 1:length(monitors)))

"""
    OptHistory

History of an optimisation, one entry per accepted iteration (vectors): `iteration`, `objective` (scaled value at
the accepted design), `kkt` (optimality measure, `NaN` if the optimiser does not compute it), `stationarity`
(stationarity measure of the design before the step, see [`StopCriteria`](@ref); `NaN` if the optimiser does not
compute it), `change` (`maximum(abs, Δz)`), `step` (step size or move limit, the mean over the entries with adaptive
limits), `attempts`, `forward`, `adjoint`, `filter` (solves in the iteration); `constraints`, `violations`,
`multipliers` and `penalties` (named tuples of vectors, one per constraint of the [`Optimization`](@ref): scaled value
and violation at the accepted design, multiplier, and penalty of the augmented Lagrangian; the multiplier of a
constraint the optimiser treats is the one it reports and its penalty is `NaN`, the multiplier of one in the augmented
Lagrangian is its `λ` before the update at the end of the iteration), `monitors` (named tuple of vectors, one per monitor of `solve!`) and
`continuation` (named tuple of vectors, one per continued parameter: its value at each iteration).
"""
mutable struct OptHistory
  n::Int
  iteration::Vector{Int}
  objective::Vector{Float64}
  kkt::Vector{Float64}
  stationarity::Vector{Float64}
  change::Vector{Float64}
  step::Vector{Float64}
  attempts::Vector{Int}
  forward::Vector{Int}
  adjoint::Vector{Int}
  filter::Vector{Int}
  constraint_names::Vector{Symbol}
  constraint_values::Matrix{Float64}    # one row per constraint
  multiplier_values::Matrix{Float64}
  violation_values::Matrix{Float64}
  penalty_values::Matrix{Float64}
  monitor_names::Vector{Symbol}
  monitor_values::Matrix{Float64}       # one row per monitor
  continuation_names::Vector{Symbol}    # continued parameters
  continuation_values::Matrix{Float64}  # one row per continued parameter
end

function OptHistory(cap::Integer, names = (); constraints = (), monitors = ())
  f() = fill(NaN, cap)
  i() = zeros(Int, cap)
  m(k) = fill(NaN, length(k), cap)
  OptHistory(0, i(), f(), f(), f(), f(), f(), i(), i(), i(), i(), collect(Symbol, constraints), m(constraints),
             m(constraints), m(constraints), m(constraints), collect(Symbol, monitors), m(monitors), collect(Symbol, names),
             m(names))
end

Base.length(h::OptHistory) = h.n

_rows(names, M, n) = NamedTuple{Tuple(names)}(Tuple(M[j, 1:n] for j in eachindex(names)))

function Base.getproperty(h::OptHistory, s::Symbol)
  s === :constraints && return _rows(getfield(h, :constraint_names), getfield(h, :constraint_values), getfield(h, :n))
  s === :multipliers && return _rows(getfield(h, :constraint_names), getfield(h, :multiplier_values), getfield(h, :n))
  s === :violations && return _rows(getfield(h, :constraint_names), getfield(h, :violation_values), getfield(h, :n))
  s === :penalties && return _rows(getfield(h, :constraint_names), getfield(h, :penalty_values), getfield(h, :n))
  s === :monitors && return _rows(getfield(h, :monitor_names), getfield(h, :monitor_values), getfield(h, :n))
  s === :continuation && return _rows(getfield(h, :continuation_names), getfield(h, :continuation_values), getfield(h, :n))
  getfield(h, s)
end
Base.propertynames(h::OptHistory) =
  (HISTORY_COLUMNS..., :constraints, :violations, :multipliers, :penalties, :monitors, :continuation)

Base.show(io::IO, h::OptHistory) =
  print(io, "OptHistory(", h.n, " iterations; constraints: ", join(h.constraint_names, ", "), "; monitors: ",
        join(h.monitor_names, ", "), ")")

# row = (iteration, objective, kkt, stationarity, change, step, attempts, forward, adjoint, filter); cons, mults, mons
# and cont: one value per constraint, constraint, monitor and continued parameter; viols and pens: one per constraint
# (empty: NaN)
function _record!(h::OptHistory, row, cons, mults, mons, cont, viols = (), pens = ())
  k = h.n += 1
  h.iteration[k], h.objective[k], h.kkt[k], h.stationarity[k], h.change[k], h.step[k], h.attempts[k], h.forward[k],
    h.adjoint[k], h.filter[k] = row
  for (j, v) in enumerate(cons)
    h.constraint_values[j, k] = v
  end
  for (j, v) in enumerate(mults)
    h.multiplier_values[j, k] = v
  end
  for (j, v) in enumerate(viols)
    h.violation_values[j, k] = v
  end
  for (j, v) in enumerate(pens)
    h.penalty_values[j, k] = v
  end
  for (j, v) in enumerate(mons)
    h.monitor_values[j, k] = v
  end
  for (j, v) in enumerate(cont)
    h.continuation_values[j, k] = v
  end
  h
end

function _trim!(h::OptHistory)
  n = h.n
  for f in HISTORY_COLUMNS
    resize!(getfield(h, f), n)
  end
  h.constraint_values = h.constraint_values[:, 1:n]
  h.multiplier_values = h.multiplier_values[:, 1:n]
  h.violation_values = h.violation_values[:, 1:n]
  h.penalty_values = h.penalty_values[:, 1:n]
  h.monitor_values = h.monitor_values[:, 1:n]
  h.continuation_values = h.continuation_values[:, 1:n]
  h
end

"""
    OptimizationFailure(msg)

Thrown by [`solve!`](@ref) when the optimisation cannot continue: the forward problem does not converge at the
current design (first iteration or right after a continuation jump), every attempt of a step was rejected, or a
gradient is not finite (failed adjoint). `z`, the fields and the states are left at the last accepted design, and
the checkpoint (if any) is written first. `history` holds the [`OptHistory`](@ref) up to the last accepted
iteration and `iterations` its number (`nothing` and `0` when thrown outside the loop).
"""
struct OptimizationFailure <: Exception
  msg::String
  history::Union{Nothing,OptHistory}
  iterations::Int
end
OptimizationFailure(msg::AbstractString) = OptimizationFailure(String(msg), nothing, 0)
Base.showerror(io::IO, e::OptimizationFailure) = print(io, "OptimizationFailure: ", e.msg)

"""
    OptResult

Result of [`solve!`](@ref): `status` (`:converged`, `:maxiter` or `:stalled`, the last when the optimiser found no
acceptable step in the last continuation stage; in an earlier stage this ends the stage), `reason` (the criterion
met: the name of its keyword in [`StopCriteria`](@ref), `:custom` for a stop function, or `nothing`), `iterations`, `objective` (scaled, at the final
design), `z` (the [`OptVariables`](@ref) of the problem, at the final design, with the fields and the forward state in
sync with it: `get_values(result.z)`), `history` ([`OptHistory`](@ref)), `time` (seconds of the call), `state` (the
state of the optimiser, for advanced use) and `start_scaling` (the factor `t` by which an infeasible starting design
was scaled towards the lower bound, `nothing` if it was not) and `multipliers` (named tuple, one per constraint: the
last multiplier reported by the optimiser for a native constraint, the final `λ` of the augmented Lagrangian, after
its last update, for the others). The full state of the run is in `problem.state[]` ([`OptState`](@ref)).
"""
struct OptResult{Z,S}
  status::Symbol
  reason::Union{Nothing,Symbol}
  iterations::Int
  objective::Float64
  z::Z
  history::OptHistory
  time::Float64
  state::S
  start_scaling::Union{Nothing,Float64}
  multipliers::NamedTuple
end

Base.show(io::IO, r::OptResult) =
  print(io, "OptResult(:", r.status, r.reason === nothing ? "" : " ($(r.reason))", ", ", r.iterations,
        " iterations, objective ", @sprintf("%.6e", r.objective), ", ", @sprintf("%.2f", r.time), " s)")

# ---------------------------------------------------------------------------------------------------------------
# Checkpoint files

function _atomic_write(f, file)
  tmp = file * ".tmp"
  open(f, tmp, "w")
  mv(tmp, file; force = true)
end

# the history as CSV: the base columns, then `constraint:<name>`, `multiplier:<name>`, `monitor:<name>`, one column per
# continued parameter, `violation:<name>` and `penalty:<name>`
function _write_history(io, hist::OptHistory)
  cn = hist.constraint_names
  head = vcat(String[String(c) for c in HISTORY_COLUMNS], ["constraint:$c" for c in cn], ["multiplier:$c" for c in cn],
              ["monitor:$m" for m in hist.monitor_names], String[String(c) for c in hist.continuation_names],
              ["violation:$c" for c in cn], ["penalty:$c" for c in cn])
  println(io, join(head, ","))
  for j in 1:hist.n
    row = Any[getfield(hist, f)[j] for f in HISTORY_COLUMNS]
    append!(row, view(hist.constraint_values, :, j))
    append!(row, view(hist.multiplier_values, :, j))
    append!(row, view(hist.monitor_values, :, j))
    append!(row, view(hist.continuation_values, :, j))
    append!(row, view(hist.violation_values, :, j))
    append!(row, view(hist.penalty_values, :, j))
    println(io, join(string.(row), ","))
  end
end

function _read_history(file)
  lines = readlines(file)
  header = split(first(lines), ",")
  rows = [parse.(Float64, split(l, ",")) for l in lines[2:end]]
  n = length(rows)
  col(j) = [r[j] for r in rows]
  pre(p) = [(j, Symbol(h[length(p)+1:end])) for (j, h) in enumerate(header) if startswith(h, p)]
  cons, mults, mons = pre("constraint:"), pre("multiplier:"), pre("monitor:")
  viols, pens = pre("violation:"), pre("penalty:")
  nbase = length(HISTORY_COLUMNS)
  conts = [(j, Symbol(header[j])) for j in nbase+1:length(header) if !occursin(':', header[j])]
  Tuple(Symbol.(header[1:nbase])) == HISTORY_COLUMNS ||
    throw(ArgumentError("load_checkpoint: the history has columns $(join(header[1:nbase], ", "))"))
  h = OptHistory(n, last.(conts); constraints = last.(cons), monitors = last.(mons))
  for k in 1:n
    r = rows[k]
    _record!(h, (Int(r[1]), r[2], r[3], r[4], r[5], r[6], Int(r[7]), Int(r[8]), Int(r[9]), Int(r[10])),
             (r[j] for (j, _) in cons), (r[j] for (j, _) in mults), (r[j] for (j, _) in mons), (r[j] for (j, _) in conts),
             (r[j] for (j, _) in viols), (r[j] for (j, _) in pens))
  end
  h
end

# a call of the post-processor at the loop's design (synced if a trial or a constraint correction left other values)
function _postprocess!(post::OptPostProcessor, oracle, hist)
  ctx = oracle.ctx
  get_values(oracle.z) == oracle.zk || _sync!(oracle)
  set_context!(ctx, :phase, :optimization)
  _post_call!(post, ctx)
  post.filepath !== nothing && post.history &&
    _atomic_write(io -> _write_history(io, hist), joinpath(post.filepath, "history.csv"))
  nothing
end

"""
    OptState

The state of an optimisation by name, what a [`Checkpoint`](@ref) writes and `resume` reads (also kept in
`problem.state[]` by `solve!`, so `resume = problem.state[]` continues without files): `iteration`, `stage`,
`stage_iterations`, `z` (the last accepted design), `blocks` and `frozen_cells` (by block), `continuation` (names),
`scales` (named tuple: `objective` and one per constraint), `history` ([`OptHistory`](@ref)), `method` and `optimizer`
(names), `memory` (of the step, see [`memory`](@ref HyperShape.Optimizers.memory); `nothing` after a failure),
`handled` (for each constraint, `:native` or `:lagrangian`), `lagrangian` (`nothing`, or the multipliers, penalties,
penalty references and number of updates of the augmented Lagrangian, by name) and `linear_offsets` (the offset `V`
of each linear constraint, kept so that a resumed run uses the same bits). Nothing in it is tied to the mesh except
the vectors of the design (`z` and those of `memory`).
"""
struct OptState
  iteration::Int
  stage::Int
  stage_iterations::Int
  z::Vector{Float64}
  blocks::Vector{Symbol}
  frozen_cells::Dict{Symbol,Int}
  continuation::Vector{Symbol}
  scales::NamedTuple
  history::OptHistory
  method::String
  optimizer::String
  memory::Union{Nothing,NamedTuple}
  handled::Dict{Symbol,Symbol}
  lagrangian::Union{Nothing,NamedTuple}
  linear_offsets::Dict{Symbol,Float64}
end

Base.show(io::IO, s::OptState) =
  print(io, "OptState(iteration ", s.iteration, ", stage ", s.stage, ", ", length(s.z), " variables, ", s.method, ")")

# `opt_state === nothing` (a failure): the memory of the optimiser is not kept, a resumed run starts from `init`
function _capture(problem, method, opt, opt_state, zk, hist, k, stage, stage_iters, cont_names, oracle)
  names = collect(keys(problem.constraints))
  al = oracle isa AugmentedLagrangianOracle ? oracle : nothing
  fo = _fo(oracle)
  mem = opt_state === nothing ? nothing : memory(opt, opt_state, zk)
  scn = (:objective, names...)
  scales = NamedTuple{scn}((get_scale(problem.objective), (get_scale(c.F) for c in problem.constraints)...))
  OptState(k, stage, stage_iters, _gather_vec(zk), collect(problem.z.names),
           Dict{Symbol,Int}(n => (b.zone === nothing ? 0 : b.zone.ncells) for (n, b) in zip(problem.z.names, problem.z.blocks)),
           collect(Symbol, cont_names), scales, _trim!(deepcopy(hist)), method_label(method), string(nameof(typeof(opt))),
           mem === nothing ? nothing : map(x -> x isa Real ? Float64(x) : _gather_vec(x), mem),
           Dict{Symbol,Symbol}(n => (al === nothing || i in al.native ? :native : :lagrangian) for (i, n) in enumerate(names)),
           al === nothing ? nothing :
             (multipliers = Dict{Symbol,Float64}(n => al.λ[i] for (i, n) in enumerate(names)),
              penalties = Dict{Symbol,Float64}(n => al.μ[i] for (i, n) in enumerate(names)),
              penalty_reference = Dict{Symbol,Float64}(n => al.vref[i] for (i, n) in enumerate(names)),
              updates = al.updates),
           Dict{Symbol,Float64}(n => fo.V[i] for (i, n) in enumerate(names) if fo.a[i] !== nothing))
end
_gather_vec(x) = copyto!(Vector{Float64}(undef, length(x)), x)

function _write_checkpoint(c::Checkpoint, st::OptState)
  mkpath(c.filepath)
  k = st.iteration
  zfile, hfile, mfile = "z_$k.bin", "history_$k.csv", "memory_$k.bin"
  _atomic_write(io -> write(io, st.z), joinpath(c.filepath, zfile))
  _atomic_write(io -> _write_history(io, st.history), joinpath(c.filepath, hfile))
  meta = Dict{String,Any}(
    "hypershape_checkpoint" => _CHECKPOINT_VERSION,
    "iteration" => k,
    "stage" => st.stage,
    "stage_iterations" => st.stage_iterations,
    "length" => length(st.z),
    "blocks" => String.(st.blocks),
    "frozen_cells" => Dict{String,Any}(String(n) => v for (n, v) in st.frozen_cells),
    "continuation" => String.(st.continuation),
    "scales" => Dict{String,Any}(String(n) => v for (n, v) in pairs(st.scales)),
    "z_file" => zfile,
    "history_file" => hfile,
    "method" => st.method,
    "optimizer" => st.optimizer,
    "handled" => Dict{String,Any}(String(n) => String(v) for (n, v) in st.handled),
    "linear_offsets" => Dict{String,Any}(String(n) => v for (n, v) in st.linear_offsets))
  if st.memory !== nothing
    # the scalars (Float64, in this order) and then the vectors, each of `length` values: NaN and Inf stay exact
    sc = [String(n) for (n, v) in pairs(st.memory) if v isa Real]
    vs = [String(n) for (n, v) in pairs(st.memory) if !(v isa Real)]
    _atomic_write(joinpath(c.filepath, mfile)) do io
      for n in sc; write(io, Float64(st.memory[Symbol(n)])); end
      for n in vs; write(io, st.memory[Symbol(n)]); end
    end
    meta["memory"] = Dict{String,Any}("file" => mfile, "scalars" => sc, "vectors" => vs)
  end
  if st.lagrangian !== nothing
    meta["multipliers"] = Dict{String,Any}(String(n) => v for (n, v) in st.lagrangian.multipliers)
    meta["penalties"] = Dict{String,Any}(String(n) => v for (n, v) in st.lagrangian.penalties)
    meta["penalty_reference"] = Dict{String,Any}(String(n) => v for (n, v) in st.lagrangian.penalty_reference)
    meta["lagrangian_updates"] = st.lagrangian.updates
  end
  meta["discretization"] = Dict{String,Any}("kind" => "FixedMesh", "events" => 0)   # reserved
  _atomic_write(io -> TOML.print(io, meta), joinpath(c.filepath, "checkpoint.toml"))
  for f in readdir(c.filepath)
    (occursin(r"^(z|memory)_\d+\.bin$", f) || occursin(r"^history_\d+\.csv$", f)) && f ∉ (zfile, hfile, mfile) &&
      rm(joinpath(c.filepath, f); force = true)
  end
end

const _CHECKPOINT_VERSION = 3

# the state of the run into problem.state[] and, with a checkpoint, to its files
function _save_state!(checkpoint, problem, method, opt, opt_state, zk, hist, k, stage, stage_iters, cont_names, oracle)
  st = _capture(problem, method, opt, opt_state, zk, hist, k, stage, stage_iters, cont_names, oracle)
  problem.state[] = st
  checkpoint === nothing || _write_checkpoint(checkpoint, st)
  st
end

"""
    load_checkpoint(filepath) -> OptState

Reads a checkpoint written by [`Checkpoint`](@ref) into an [`OptState`](@ref); `solve!(…; resume = filepath)` calls
it. OC (adaptive move limits, correction shift) and SiMPL (latent variable, Barzilai–Borwein memory, multiplier,
correction shift) save their memory, so a resumed run continues as the uninterrupted one; a checkpoint written after
a failure does not save the memory. Checkpoints of other format versions are refused.
"""
function load_checkpoint(filepath::AbstractString)
  file = joinpath(filepath, "checkpoint.toml")
  isfile(file) || throw(ArgumentError("load_checkpoint: no checkpoint.toml in $filepath"))
  meta = TOML.parsefile(file)
  v = get(meta, "hypershape_checkpoint", nothing)
  v == _CHECKPOINT_VERSION ||
    throw(ArgumentError("load_checkpoint: $filepath has the format $(v === nothing ? "of no version" : "version $v"); " *
                        "this version of HyperShape reads version $_CHECKPOINT_VERSION"))
  n = meta["length"]
  z = Vector{Float64}(undef, n)
  open(io -> read!(io, z), joinpath(filepath, meta["z_file"]))
  mem = nothing
  if haskey(meta, "memory")
    m = meta["memory"]
    sc, vs = Symbol.(m["scalars"]), Symbol.(m["vectors"])
    vals = open(joinpath(filepath, m["file"])) do io
      (Tuple(read(io, Float64) for _ in sc)..., Tuple(read!(io, Vector{Float64}(undef, n)) for _ in vs)...)
    end
    mem = NamedTuple{(sc..., vs...)}(vals)
  end
  sc = meta["scales"]
  names = (:objective, (Symbol(k) for k in sort!(collect(keys(sc))) if k != "objective")...)
  scales = NamedTuple{names}(Tuple(Float64(sc[String(k)]) for k in names))
  OptState(Int(meta["iteration"]), Int(meta["stage"]), Int(meta["stage_iterations"]), z, Symbol.(meta["blocks"]),
           Dict{Symbol,Int}(Symbol(k) => Int(v) for (k, v) in get(meta, "frozen_cells", Dict())),
           Symbol.(meta["continuation"]), scales, _read_history(joinpath(filepath, meta["history_file"])),
           get(meta, "method", ""), get(meta, "optimizer", ""), mem,
           Dict{Symbol,Symbol}(Symbol(k) => Symbol(v) for (k, v) in get(meta, "handled", Dict())),
           haskey(meta, "multipliers") ?
             (multipliers = _by_name(meta["multipliers"]), penalties = _by_name(meta["penalties"]),
              penalty_reference = _by_name(meta["penalty_reference"]), updates = Int(meta["lagrangian_updates"])) : nothing,
           _by_name(get(meta, "linear_offsets", Dict())))
end

_by_name(d) = Dict{Symbol,Float64}(Symbol(k) => Float64(v) for (k, v) in d)

# the rows of a loaded history into the history of the run (same constraints; a monitor not in the checkpoint is NaN)
function _prefill!(hist::OptHistory, old::OptHistory)
  mon = [findfirst(==(m), old.monitor_names) for m in hist.monitor_names]
  for k in 1:old.n
    row = (old.iteration[k], old.objective[k], old.kkt[k], old.stationarity[k], old.change[k], old.step[k],
           old.attempts[k], old.forward[k], old.adjoint[k], old.filter[k])
    _record!(hist, row, view(old.constraint_values, :, k), view(old.multiplier_values, :, k),
             (j === nothing ? NaN : old.monitor_values[j, k] for j in mon), view(old.continuation_values, :, k),
             view(old.violation_values, :, k), view(old.penalty_values, :, k))
  end
  hist
end

# ---------------------------------------------------------------------------------------------------------------
# Solve counters

function _counters(problem)
  Fs = (problem.objective, (c.F for c in problem.constraints)...)
  fwd, filters = Any[], Any[]
  for F in Fs
    for S in responses(F)
      any(o -> o === S, fwd) || push!(fwd, S)
    end
    for op in filter_operators(physical_fields(F))
      any(o -> o === op, filters) || push!(filters, op)
    end
  end
  Tuple(fwd), Tuple(S.adjoint for S in fwd), Tuple(filters)
end

_count(ops) = sum(num_solves, ops; init = 0)

# ---------------------------------------------------------------------------------------------------------------
# The loop

function _stop_reason(stop::StopCriteria, info::StepInfo, hist::OptHistory, stage_start)
  if stop.objective_range !== nothing && hist.n - stage_start >= stop.window
    w = view(hist.objective, hist.n-stop.window+1:hist.n)
    lo, hi = extrema(w)
    d = max(maximum(abs, w), 1e-3 * abs(hist.objective[1]))   # an objective tending to zero: floor from the first
    (hi - lo) <= stop.objective_range * d && return :objective_range
  end
  hist.n - stage_start >= 2 || return nothing   # the first iteration of a stage (restoration after a jump)
  stop.kkt_tol !== nothing && info.kkt < stop.kkt_tol && return :kkt_tol
  stop.stationarity_tol !== nothing && info.stationarity < stop.stationarity_tol && return :stationarity_tol
  if stop.stationarity_rtol !== nothing
    s1 = _first_finite(hist.stationarity, hist.n)
    isfinite(s1) && s1 > 0 && info.stationarity < stop.stationarity_rtol * s1 && return :stationarity_rtol
  end
  if stop.kkt_rtol !== nothing
    k1 = hist.kkt[1]
    isfinite(k1) && k1 > 0 && info.kkt < stop.kkt_rtol * k1 && return :kkt_rtol
  end
  stop.change_tol !== nothing && info.change < stop.change_tol && return :change_tol
  nothing
end
_stop_reason(::Nothing, info, hist, stage_start) = nothing
# a function of the user: (info, history, ctx) -> Bool, checked after every accepted iteration (also the first of a
# stage; with a continuation, only in the last stage unless close_early); the reason is :custom
function _stop_reason(stop, info, hist, stage_start, ctx)
  r = stop(info, _trim!(deepcopy(hist)), ctx)        # the history up to this iteration (a copy: O(iterations))
  r isa Bool || throw(ArgumentError("solve!: the stop function must return true or false, got $(repr(r))"))
  r ? :custom : nothing
end
_stop_reason(stop::Union{Nothing,StopCriteria}, info, hist, stage_start, ctx) = _stop_reason(stop, info, hist, stage_start)

"""
    solve!(problem::Optimization, method::DensityMethod; maxiter, on_failure = RejectStep(),
           stop = default_stop(optimizer), continuation = nothing, checkpoint = nothing, post = nothing,
           monitors = (;), resume = nothing, verbose = 2, io = stdout) -> OptResult

Runs the optimisation from the current values of `z`. Mandatory: `maxiter` (total iterations, resumed ones included).
Optional: `on_failure` ([`RejectStep`](@ref), default `RejectStep(shrink = 0.5, max_attempts = 5)`), `stop`
([`StopCriteria`](@ref); default [`default_stop`](@ref): range of the objective below 0.1 % over 10 iterations of
the last stage; `stop = nothing` runs `maxiter` iterations; or a function (or callable object)
`(info, history, ctx) -> Bool` of the user, checked after every accepted iteration, with the
[`StepInfo`](@ref HyperShape.Optimizers.StepInfo) of the step (`info.f` is the objective before it), the
[`OptHistory`](@ref) up to this iteration (a copy) and the context: `true` stops with reason `:custom`, under the
same rules as `StopCriteria` (only in the last stage of a `Continuation` unless `close_early`; with a nonlinear
constraint, only when it holds to the tolerance of the optimiser)), `continuation` ([`Continuation`](@ref)), `checkpoint`
([`Checkpoint`](@ref)), `post` (an
[`OptPostProcessor`](@ref HyperShape.PostProcess.OptPostProcessor): its driver is called at the starting design, at the
accepted iterations that are multiples of `every`, at the end and after a failure, with the fields and the forward state at the accepted
design) and
`monitors` (a named tuple of monitors, such as `(grey = Grey(θ, dΩ),)`: their values, see `monitor_value`, go to the
history, the table and the closing line), `resume` (the folder of a [`Checkpoint`](@ref) or the result of
[`load_checkpoint`](@ref)), `verbose` and `io`. An `OC` whose `move` is continued by `continuation` gets its own
`move` back when `solve!` returns or throws; the other continued parameters keep the values of the last stage.

Screen output (`verbose`, default `2`, printed to `io`): `0` nothing; `1` a banner (variables, constraint, metric,
continuation, stopping criteria and failure handling, each marked when it is a default), the events (continuation
jumps, rejected steps, stalls, a scaled start), a closing
✓/⚠/✗ line and a summary of solves and time; `2` also a row per iteration: objective, its relative change,
constraint (red above 1e-3), optimality measure `kkt`, stationarity measure `S`, change of z, step, multiplier,
continued parameters, monitors, forward, adjoint and filter solves, and time. `true` is `2`, `false` is `0`. The summary counts every solve of the call
(those of rejected or stalled trials, the setup and the final synchronisation included).

With a `checkpoint`, an exception that is not an [`OptimizationFailure`](@ref) (an error in a solve, post or monitor
function of the user, an interruption) writes a checkpoint at the last accepted design (history included) and is then
rethrown; an `OptimizationFailure` writes its own.

If the starting design (not a resumed one) violates the constraint, it is scaled towards the lower bound,
`z = lb + t (z − lb)` on the entries the constraint acts on (a linear constraint: so that it holds with equality; a
nonlinear one: with `t` found by regula falsi so that `-1e-6 ≤ g ≤ 0`); `t` is returned in `result.start_scaling`
and reported as an event when `verbose ≥ 1`. At a continuation jump the design is not touched: an optimiser that
finds it infeasible restores feasibility in its next step.

Each iteration: the optimiser takes a step from the current design; the loop evaluates the new design (no solve when
the optimiser already did, as SiMPL); if the forward problem does not converge there, the design is restored, the
step shrunk and tried again. Solves per iteration: OC one filter, one forward, one adjoint (none for a self-adjoint
functional) and one transposed filter; SiMPL one filter and one forward per trial of its line search (usually one),
one adjoint and one transposed filter; one more forward after each continuation jump. A linear constraint costs
nothing in the loop (its vector is computed once at the start); a nonlinear one costs, per step, one transposed filter
for its derivative and one filter per correction (see `OC` and `SiMPL`), and no forward solve. With a nonlinear
constraint the run only converges when the constraint also holds to the `constraint_tol` of the optimiser.

With several constraints, or one the optimiser does not treat itself, the method's augmented Lagrangian
([`DensityMethod`](@ref)`(…; lagrangian)`) takes the others: the banner says who treats each one, the table shows the
largest violation and the largest `|λ|`, the multipliers are updated after every accepted iteration (no solve: the
values at the accepted design are cached) and the run converges only when every constraint holds to its tolerance (a
run that ends otherwise says which constraints do not). The scaled start acts on the native constraint only. A
`stop` function receives the `StepInfo` of the optimiser, whose `f` is then the augmented objective; the history
keeps the objective of the problem. Per iteration, the constraints of the augmented Lagrangian add no forward solve
and, with their states, one adjoint solve per state in the same reverse sweep as the objective.
"""
function solve!(problem::Optimization, method::AbstractMethod; maxiter::Integer, on_failure = _DefaultFailure(),
                stop = _DefaultStop(), continuation = nothing, checkpoint = nothing, post = nothing, monitors = (;),
                resume = nothing, verbose = 2, io::IO = stdout)
  t_call = time()
  schedule = continuation
  z, ctx, opt = problem.z, problem.ctx, stepper(method)
  monitors isa NamedTuple || throw(ArgumentError("solve!: monitors must be a named tuple, (grey = Grey(θ, dΩ),)"))
  for (n, m) in pairs(monitors)
    applicable(monitor_value, m) || throw(ArgumentError("solve!: monitor $n: no monitor_value method for $(typeof(m))"))
  end
  resume isa AbstractString && (resume = load_checkpoint(resume))
  resume === nothing || resume isa OptState ||
    throw(ArgumentError("solve!: resume must be the folder of a Checkpoint, the result of load_checkpoint or problem.state[]"))
  defaults = (stop = stop isa _DefaultStop, on_failure = on_failure isa _DefaultFailure)
  defaults.stop && (stop = default_stop(opt))
  defaults.on_failure && (on_failure = RejectStep())
  on_failure isa RejectStep || throw(ArgumentError("solve!: on_failure must be a RejectStep"))
  stop === nothing || stop isa StopCriteria || !isempty(methods(stop)) ||
    throw(ArgumentError("solve!: stop must be a StopCriteria, nothing or a function (info, history, ctx) -> Bool"))
  schedule === nothing || schedule isa Continuation || throw(ArgumentError("solve!: continuation must be a Continuation"))
  schedule === nothing || all(o -> !(o isa Optimizer) || o === opt, schedule.objects) ||
    throw(ArgumentError("solve!: a Continuation of an optimiser must act on the one of the method"))
  checkpoint === nothing || checkpoint isa Checkpoint || throw(ArgumentError("solve!: checkpoint must be a Checkpoint"))
  post === nothing || post isa OptPostProcessor ||
    throw(ArgumentError("solve!: post must be an OptPostProcessor, for example OptPostProcessor((post, ctx) -> …; every = 1)"))
  maxiter >= 1 || throw(ArgumentError("solve!: maxiter must be at least 1"))
  if stop isa StopCriteria && (stop.kkt_tol !== nothing || stop.kkt_rtol !== nothing) && !computes_kkt(opt)
    throw(ArgumentError("StopCriteria: kkt needs an optimiser that computes an optimality measure ($(nameof(typeof(opt))) does not)"))
  end
  taken = (HISTORY_COLUMNS..., :constraint, :μ, :λ, :S, (schedule === nothing ? () : schedule.names)...)
  for n in keys(monitors)
    n in taken && throw(ArgumentError("solve!: $n is the name of a column of the table or a continued parameter; " *
                                      "choose another name for the monitor"))
  end
  check_problem(method, problem)
  native = _native_split(problem, opt, augmented_lagrangian(method))
  pr = _Printer(io, verbose, schedule, keys(monitors), length(problem.constraints) > 1)
  # a nonlinear constraint is met to the tolerance of the optimiser: above it, the value is shown in red
  # a constraint violated beyond its tolerance is shown in red (set once the split is known; see below)
  cont_names = schedule === nothing ? () : schedule.names
  Fs = (problem.objective, (c.F for c in problem.constraints)...)

  k0, stage, stage_iters = 0, schedule === nothing ? 0 : 1, 0
  cnames = keys(problem.constraints)
  hist = OptHistory(maxiter, cont_names; constraints = cnames, monitors = keys(monitors))
  if resume !== nothing
    length(resume.z) == length(z) || throw(ArgumentError("solve!: the checkpoint has $(length(resume.z)) values, z has $(length(z))"))
    resume.blocks == collect(z.names) || throw(ArgumentError("solve!: the checkpoint has blocks $(resume.blocks), z has $(collect(z.names))"))
    for (n, b) in zip(z.names, z.blocks)
      nc = b.zone === nothing ? 0 : b.zone.ncells
      get(resume.frozen_cells, n, nc) == nc || throw(ArgumentError("solve!: the checkpoint has $(resume.frozen_cells[n]) " *
                                                                  "frozen cells in $n, z has $nc"))
    end
    resume.continuation == collect(cont_names) ||
      throw(ArgumentError("solve!: the checkpoint continues $(resume.continuation), the continuation $(collect(cont_names))"))
    nst = schedule === nothing ? 0 : schedule.nstages
    (schedule === nothing ? resume.stage == 0 : 1 <= resume.stage <= nst) ||
      throw(ArgumentError("solve!: the checkpoint is at stage $(resume.stage); the continuation has $nst stages"))
    Set(keys(resume.scales)) == Set((:objective, cnames...)) ||
      throw(ArgumentError("solve!: the checkpoint has functionals $(join(keys(resume.scales), ", ")); the problem has " *
                          join((:objective, cnames...), ", ")))
    resume.iteration < maxiter || throw(ArgumentError("solve!: the checkpoint is already at iteration $(resume.iteration) ≥ maxiter"))
    set_values!(z, resume.z)
    # a scale fixed by :initial is restored; a scale given by the user stays
    foreach((F, n) -> (F.scale_mode === :initial && (F.scale[] = getproperty(resume.scales, n))), Fs, (:objective, cnames...))
    k0, stage, stage_iters = resume.iteration, resume.stage, resume.stage_iterations
    _prefill!(hist, resume.history)
    dropped = setdiff(resume.history.monitor_names, collect(keys(monitors)))
    isempty(dropped) || _print_event(pr, "· monitors of the checkpoint not given now (their columns are dropped): " *
                                         join(dropped, ", "), :yellow)
  end

  # an optimiser whose parameters the schedule continues gets them back at the end (the stages change them)
  restore_opt = _restorer(opt)
  try
  # the parameters of the first stage are applied before the banner, which shows them
  cont_buf = zeros(length(cont_names))
  set_context!(ctx, :continuation, schedule === nothing ? NamedTuple() : _apply_stage!(schedule, stage, cont_buf))
  _print_banner(pr, problem, method, maxiter, stop, on_failure, defaults, schedule, resume, native)
  set_context!(ctx, :iteration, k0)
  set_context!(ctx, :attempt, 0)
  set_context!(ctx, :trial, 0)

  fo = build_oracle(method, problem)
  ops = _counters(problem)
  cstart = map(_count, ops)
  for i in 1:num_constraints(fo)
    _is_linear(fo, i) && _setup_constraint!(fo, i, ctx)
  end
  # the offsets of the linear constraints of the run that is resumed (computed again they could differ in the last bit)
  if resume !== nothing
    for (i, n) in enumerate(keys(problem.constraints))
      fo.a[i] === nothing || (fo.V[i] = get(resume.linear_offsets, n, fo.V[i]))
    end
  end
  set_context!(ctx, :phase, :optimization)
  zk = fo.zk
  copyto!(zk, get_values(z))
  resume === nothing || _check_handled(native, resume, cnames)
  t_start = nothing
  if resume === nothing                       # a resumed design is the one the run left
    t_start = isempty(native) ? nothing : _make_feasible!(fo, zk, native[1])
    # a start feasible up to rounding (t within 1e-6 of 1) is not worth an event
    (t_start === nothing || t_start > 1 - 1e-6) || _print_event(pr, "· the starting design violated the constraint; it was scaled " *
                                            "towards the lower bound by $(round(t_start, sigdigits = 4)) to meet it", :yellow)
  end
  # after the feasible start: a scale = :initial of a constraint is taken at the design the run starts from
  oracle = length(native) == num_constraints(fo) ? fo : AugmentedLagrangianOracle(fo, augmented_lagrangian(method), native, zk)
  oracle isa AugmentedLagrangianOracle && resume !== nothing && _restore_lagrangian!(oracle, resume, cnames)
  pr.tols = [i in native ? (is_linear(problem.constraints[i].F) ? _RED_CONSTRAINT : min(_RED_CONSTRAINT, constraint_tol(opt))) :
             augmented_lagrangian(method).constraint_tol for i in 1:length(problem.constraints)]
  zprev = similar(zk)
  state = init(opt, zk, oracle)
  if resume !== nothing && resume.memory !== nothing
    if resume.optimizer == string(nameof(typeof(opt)))
      restore_memory!(opt, state, zk, resume.memory)
    else
      _print_event(pr, "· the memory of the optimiser was not restored (checkpoint written by $(resume.optimizer))", :yellow)
    end
  end
  if post !== nothing
    _post_start!(post, problem, k0, resume !== nothing)
    # a resumed run that starts with a continuation jump: the design of k0 is the one of the previous stage, and
    # solving it now would cost one more forward solve (the first iteration solves it again at the new stage)
    jump_first = schedule !== nothing && stage < schedule.nstages && stage_iters >= schedule.every
    if post.initial && post.last != k0 && !jump_first
      # a resumed run writes k0 again: files without a collection entry (VTKHDF) keep only the steps before it
      resume !== nothing && (post.resume_upto = k0 - 1)
      # the forward solve of the first iteration, done now (the first step takes it from the value cache)
      if isfinite(_sync!(oracle))
        _postprocess!(post, oracle, hist)
      else
        post.resume_upto = resume === nothing ? nothing : k0      # k0 is not written again: keep it
        _print_event(pr, "· the forward problem does not converge at the starting design: it is not post-processed", :yellow)
      end
    end
  end
  fwd, adj, filters = ops
  run = _LoopState(k0, stage, stage_iters, hist.n - stage_iters, cstart, k0, copy(zk))   # what a failure needs
  status, reason = try
    _iterate!(problem, method, opt, oracle, state, zk, zprev, hist, run, ctx, cont_buf, cont_names, maxiter, on_failure, stop,
              schedule, checkpoint, post, monitors, fwd, adj, filters, pr)
  catch e
    # OptimizationFailure has written its checkpoint; any other exception (an error in a solve or post function of
    # the user, an interruption) gets one at the last accepted design before it is rethrown
    if !(e isa OptimizationFailure)
      try
        _save_state!(checkpoint, problem, method, opt, nothing, run.zacc, hist, run.k, run.stage, run.stage_iters, cont_names, oracle)
        checkpoint === nothing ||
          _print_event(pr, "✗ $(nameof(typeof(e))) after $(run.k) accepted iterations: checkpoint written at iteration " *
                           "$(run.k) in $(checkpoint.filepath)", :red)
      catch e2
        e2 isa InterruptException || _print_event(pr, "✗ the checkpoint could not be written after the exception", :red)
      end
    end
    rethrow()
  end

  f_final = _sync!(oracle)                     # fields and forward state at the final design
  _print_end(pr, status, reason, run.k, f_final, _last_monitors(hist, monitors))
  status === :converged || _warn_unmet(pr, hist, cnames)
  _print_summary(pr, map(_count, ops) .- cstart, run.k - k0)
  final = _save_state!(checkpoint, problem, method, opt, state, zk, hist, run.k, run.stage, run.stage_iters, cont_names, oracle)
  if post !== nothing && post.last != run.k && isfinite(f_final)
    set_context!(ctx, :iteration, run.k)       # after a stall, the iteration tried was run.k + 1
    _postprocess!(post, oracle, hist)
  end
  # pending writes finished; a write that fails now is reported, not thrown: the optimisation has finished and its
  # result is returned (decided by Claude, delegated by Jesús, 2026-11-16)
  post === nothing || _post_close!(post; quiet = true)
  lastm(j) = hist.n == 0 ? NaN : hist.multiplier_values[j, hist.n]
  mults = NamedTuple{Tuple(cnames)}(Tuple(final.lagrangian !== nothing && final.handled[n] === :lagrangian ?
                                          final.lagrangian.multipliers[n] : lastm(j) for (j, n) in enumerate(cnames)))
  OptResult(status, reason, run.k, f_final, z, _trim!(hist), time() - t_call, state, t_start, mults)
  finally
    restore_opt()
    post isa OptPostProcessor && _post_close!(post; quiet = true)   # failure or interruption: finish the writes
  end
end

# the optimiser's own continued parameters, given back when solve! returns or throws
_restorer(opt) = () -> nothing
_restorer(opt::OC) = (move0 = opt.move; () -> set_move!(opt, move0))

# The constraints the optimiser treats itself (native, see `handles`): `lag.native = :auto` takes, in the order of the
# problem, the first ones it handles (up to `max_native`); a tuple of names takes those. The others go to the augmented
# Lagrangian. Decided before anything is solved: whether a constraint is handled does not depend on its scale.
function _native_split(problem, opt, lag)
  infos = map(problem.constraints) do c
    r = _reference(c)
    ConstraintInfo(c.lo - r, c.hi - r; linear = is_linear(c.F), design_only = is_design_only(c.F))
  end
  names = collect(keys(problem.constraints))
  native = Int[]
  if lag.native === :auto
    for (i, info) in enumerate(infos)
      length(native) < max_native(opt) && handles(opt, info) && push!(native, i)
    end
  else
    allunique(lag.native) || throw(ArgumentError("AugmentedLagrangian: native repeats a name"))
    for n in lag.native
      i = findfirst(==(n), names)
      i === nothing && throw(ArgumentError("AugmentedLagrangian: native names $n, which is not a constraint of the problem"))
      handles(opt, infos[i]) || throw(ArgumentError("AugmentedLagrangian: $(nameof(typeof(opt))) does not treat the " *
        "constraint $n itself (see HyperShape.Optimizers.handles)"))
      push!(native, i)
    end
    length(native) <= max_native(opt) ||
      throw(ArgumentError("AugmentedLagrangian: $(nameof(typeof(opt))) treats at most $(max_native(opt)) constraint(s) itself"))
  end
  requires_native(opt) && !isempty(names) && isempty(native) &&
    throw(ArgumentError("solve!: OC needs one constraint F ≤ max that solves no state (the volume) of its own; " *
                        "use SiMPL, or keep that constraint native"))
  native
end

# a run that ends (maxiter, stall) with a constraint violated beyond its tolerance says which
function _warn_unmet(pr, hist, cnames)
  (hist.n == 0 || isempty(pr.tols)) && return nothing
  bad = [string(n) for (j, n) in enumerate(cnames) if hist.violation_values[j, hist.n] > pr.tols[j]]
  isempty(bad) || _print_event(pr, "⚠ constraints not met to their tolerance at the last design: " * join(bad, ", "), :yellow)
  nothing
end

# The augmented Lagrangian over the FunctionalOracle: what the loop asks of the oracle of the problem
_fo(o::FunctionalOracle) = o
_fo(al::AugmentedLagrangianOracle) = al.inner
_sync!(al::AugmentedLagrangianOracle) = _sync!(al.inner)
_invalidate_snapshot!(al::AugmentedLagrangianOracle) = (al.snap_valid = false; _invalidate_snapshot!(al.inner); al)
_postprocess!(post::OptPostProcessor, al::AugmentedLagrangianOracle, hist) = _postprocess!(post, al.inner, hist)
_constraint_values(al::AugmentedLagrangianOracle, zk) = _constraint_values(al.inner, zk)
# the objective of the problem at the last objective call (the augmented one adds the terms of the constraints)
_problem_objective(::FunctionalOracle, f) = f
_problem_objective(al::AugmentedLagrangianOracle, f) = al.f
# the multipliers of the native constraints from the optimiser, λ of the others
_multipliers(al::AugmentedLagrangianOracle{O}, info) where {O<:FunctionalOracle{J,C}} where {J,C} =
  ntuple(i -> (j = findfirst(==(i), al.native); j === nothing ? al.λ[i] :
               j <= length(info.multipliers) ? info.multipliers[j] : NaN), Val(fieldcount(C)))
_penalties(o::FunctionalOracle{J,C}) where {J,C} = ntuple(_ -> NaN, Val(fieldcount(C)))
_penalties(al::AugmentedLagrangianOracle{O}) where {O<:FunctionalOracle{J,C}} where {J,C} = ntuple(i -> al.μ[i], Val(fieldcount(C)))
_violations(o, vals) = ntuple(i -> violation(constraint_info(_fo(o), i), vals[i]), length(vals))
_violations(al::AugmentedLagrangianOracle, vals) = ntuple(i -> violation(al.infos[i], vals[i]), length(vals))

# with a nonlinear native constraint, convergence needs it met to the tolerance of the optimiser; the constraints of
# the augmented Lagrangian, to its tolerance
_constraints_met(o::FunctionalOracle, info, opt, viols) =
  !any(i -> o.a[i] === nothing, 1:num_constraints(o)) || violation(info) <= constraint_tol(opt)
function _constraints_met(al::AugmentedLagrangianOracle, info, opt, viols)
  fo = al.inner
  any(i -> fo.a[i] === nothing, al.native) && violation(info) > constraint_tol(opt) && return false
  all(i -> viols[i] <= al.tol, al.absorbed)
end

# multipliers and penalties of a checkpoint, by name; the split native/augmented Lagrangian must be the same
function _check_handled(native, resume, cnames)
  handled = resume.handled
  for (i, n) in enumerate(cnames)
    want = i in native ? :native : :lagrangian
    get(handled, n, want) == want || throw(ArgumentError("solve!: the checkpoint treats the constraint $n as " *
      "$(handled[n]); now it would be $want (change native in AugmentedLagrangian to match)"))
  end
end

function _restore_lagrangian!(al::AugmentedLagrangianOracle, resume, cnames)
  r = resume.lagrangian
  r === nothing && return al
  for i in al.absorbed
    n = cnames[i]
    haskey(r.multipliers, n) || throw(ArgumentError("solve!: the checkpoint has no multiplier for the constraint $n"))
    al.λ[i] = r.multipliers[n]
    al.μ[i] = r.penalties[n]
    al.vref[i] = r.penalty_reference[n]
  end
  al.updates = r.updates
  al
end

# A start that violates the constraint is scaled towards the lower bound, z = lb + t (z - lb), so that it holds (with
# equality for a linear constraint, -1e-6 ≤ g ≤ 0 for a nonlinear one): the pattern of the design is kept, the bounds
# hold, and optimisers that need a feasible start (SiMPL) can start.
function _make_feasible!(o, zk, i = 1)
  num_constraints(o) == 0 && return nothing
  o.a[i] === nothing && return _make_feasible_nonlinear!(o, zk, i)
  a, V = o.a[i], o.V[i]
  az = dot(a, zk)
  az - V <= 1e-12 * max(abs(V), 1.0) && return nothing
  alb = dot(a, o.lb)
  (all(>=(0), a) && alb < V) ||
    throw(ArgumentError("Optimization: the linear constraint cannot be met within the bounds (aᵀlb ≥ V or a has negative entries)" *
                        _zone_hint(o)))
  t = (V - alb) / (az - alb)
  @. zk = ifelse(a > 0, o.lb + t * (zk - o.lb), zk)     # entries outside the constraint are left as they are
  set_values!(o.z, zk)
  t
end

# t by regula falsi (Illinois) on g(lb + t (z - lb)) = -tol/2 in (0, 1], on the entries the constraint depends on (a
# nonzero derivative at the start); each evaluation costs the filter solves
function _make_feasible_nonlinear!(o, zk, i = 1; tol = 1e-6, maxit = 60)
  g1 = constraint(o, i, zk)
  isfinite(g1) || throw(ArgumentError("Optimization: the constraint is not finite at the starting design"))
  g1 <= 0 && return nothing
  # entries the constraint depends on: nonzero derivative at the start or at the middle of the bounds (a saturated
  # projection gives exact zeros where the constraint still depends on the entry)
  acts = constraint_gradient!(similar(zk), o, i, zk) .!= 0
  zmid = (o.lb .+ o.ub) ./ 2
  constraint(o, i, zmid)
  acts .|= constraint_gradient!(similar(zk), o, i, zmid) .!= 0
  constraint(o, i, zk)
  z0 = copy(zk)
  zt = similar(zk)
  gt(t) = (@. zt = ifelse(acts, o.lb + t * (z0 - o.lb), z0); constraint(o, i, zt) + tol / 2)
  t0, t1 = 0.0, 1.0
  h0, h1 = gt(t0), g1 + tol / 2
  h0 < 0 || throw(ArgumentError("Optimization: the constraint cannot be met within the bounds (it is violated at the lower bound)" *
                                _zone_hint(o)))
  t, side = t0, 0
  for _ in 1:maxit
    t = (t0 * h1 - t1 * h0) / (h1 - h0)
    h = gt(t)
    abs(h) <= tol / 2 && break
    if h > 0
      t1, h1 = t, h
      side == -1 && (h0 /= 2)
      side = -1
    else
      t0, h0 = t, h
      side == 1 && (h1 /= 2)
      side = 1
    end
  end
  constraint(o, i, zt) <= 0 || (t = t0; @. zt = ifelse(acts, o.lb + t * (z0 - o.lb), z0))   # end on the feasible side
  copyto!(zk, zt)
  set_values!(o.z, zk)
  t
end

_zone_hint(o) = any(b -> b.zone !== nothing, o.z.blocks) ?
  "; the frozen zones count as material: is their volume above the bound?" : ""

mutable struct _LoopState
  k::Int
  stage::Int
  stage_iters::Int
  stage_start::Int    # history length when the current stage started
  cstart::NTuple{3,Int}   # solve counters at the start of the call
  k0::Int                 # iterations done before the call (resume)
  zacc::Vector{Float64}   # the last accepted design (for a checkpoint after an unexpected exception)
end

function _fail(msg, problem, method, opt, oracle, zk, hist, run, cont_names, checkpoint, pr, ops, post)
  synced = try
    isfinite(_sync!(oracle))                   # fields and forward state at the last accepted design
  catch e
    e isa InterruptException && rethrow()
    false
  end
  synced || (msg *= " (the forward state could not be solved again at the last accepted design)")
  _print_failure(pr, msg, run.k)
  _print_summary(pr, map(_count, ops) .- run.cstart, run.k - run.k0)
  _save_state!(checkpoint, problem, method, opt, nothing, zk, hist, run.k, run.stage, run.stage_iters, cont_names, oracle)
  if synced && post !== nothing && post.last != run.k
    set_context!(problem.ctx, :iteration, run.k)
    try
      _postprocess!(post, oracle, hist)
    catch e
      e isa InterruptException && rethrow()
      _print_event(pr, "✗ the post-processing of the last accepted design threw: " * first(sprint(showerror, e), 300), :red)
    end
  end
  throw(OptimizationFailure(msg, _trim!(deepcopy(hist)), run.k))
end

function _iterate!(problem, method, opt, oracle, state, zk, zprev, hist, run, ctx, cont_buf, cont_names, maxiter, on_failure,
                   stop, schedule, checkpoint, post, monitors, fwd, adj, filters, pr)
  fail(msg) = _fail(msg, problem, method, opt, oracle, zk, hist, run, cont_names, checkpoint, pr, (fwd, adj, filters), post)
  while run.k < maxiter
    k = run.k + 1
    set_context!(ctx, :iteration, k)
    set_context!(ctx, :attempt, 1)
    set_context!(ctx, :trial, 0)
    jumped = false
    if schedule !== nothing && run.stage < schedule.nstages && run.stage_iters >= schedule.every
      jumped = true
      set_context!(ctx, :continuation, _apply_stage!(schedule, run.stage + 1, cont_buf))
      _invalidate_snapshot!(oracle)
      reset!(opt, state)
      _print_jump(pr, schedule, run.stage + 1)
    end
    copyto!(zprev, zk)
    c0 = (_count(fwd), _count(adj), _count(filters))
    info = nothing
    f_acc = NaN
    stalled = false
    while true
      info = try
        step!(opt, state, zk, oracle, ctx)
      catch e
        e isa OptimizationFailure || rethrow()
        jumped && _undo_jump!(schedule, run, ctx, cont_buf, opt, state, oracle)
        fail(e.msg)
      end
      if !info.accepted
        if !isfinite(info.f)
          jumped && _undo_jump!(schedule, run, ctx, cont_buf, opt, state, oracle)
          fail("the forward problem does not converge at the current design" *
               (k == 1 ? " (first iteration)" : jumped ? " (right after a continuation jump)" : ""))
        elseif isfinite(info.f_new)
          stalled = true
          break
        else
          jumped && _undo_jump!(schedule, run, ctx, cont_buf, opt, state, oracle)
          fail("the optimiser found no step at which the forward problem converges")
        end
      end
      f_acc = objective(oracle, zk)
      isfinite(f_acc) && (f_acc = _problem_objective(oracle, f_acc); break)
      copyto!(zk, zprev)
      if ctx.attempt >= on_failure.max_attempts
        jumped && _undo_jump!(schedule, run, ctx, cont_buf, opt, state, oracle)
        fail("the forward problem did not converge at the $(on_failure.max_attempts) designs proposed in iteration $k")
      end
      shrink!(opt, state, on_failure.shrink)
      set_context!(ctx, :attempt, ctx.attempt + 1)
      _print_event(pr, "✗ iteration $k: the forward problem did not converge at the proposed design; step shrunk by " *
                       "$(on_failure.shrink), attempt $(ctx.attempt) of $(on_failure.max_attempts)", :red)
    end
    if stalled
      jumped && (run.stage += 1; run.stage_iters = 0; run.stage_start = hist.n)
      # no step at this β: the stage has converged; go to the next one, or stop in the last
      (schedule === nothing || run.stage == schedule.nstages) && return :stalled, nothing
      _print_event(pr, "· no acceptable step in stage $(run.stage): the stage ends", :yellow)
      pr.tlast = time()                        # the time of the stalled trial is not charged to the next row
      run.stage_iters = schedule.every
      continue
    end
    if jumped
      run.stage += 1
      run.stage_iters = 0
      run.stage_start = hist.n
    end
    if stop isa StopCriteria && (stop.kkt_tol !== nothing || stop.kkt_rtol !== nothing) && isnan(info.kkt)
      throw(ArgumentError("StopCriteria: kkt needs an optimiser that computes an optimality measure ($(nameof(typeof(opt))) does not)"))
    end
    if stop isa StopCriteria && (stop.stationarity_tol !== nothing || stop.stationarity_rtol !== nothing) && isnan(info.stationarity)
      throw(ArgumentError("StopCriteria: stationarity needs an optimiser that computes it ($(nameof(typeof(opt))) " *
                          "returned NaN: it does not compute it, or its gradient is not finite)"))
    end
    run.k = k
    copyto!(run.zacc, zk)                      # with run.k: a checkpoint after an exception matches them
    run.stage_iters += 1
    c1 = (_count(fwd), _count(adj), _count(filters))
    row = (k, f_acc, info.kkt, info.stationarity, info.change, info.step, ctx.attempt, c1[1] - c0[1], c1[2] - c0[2],
           c1[3] - c0[3])
    cvals = _constraint_values(oracle, zk)
    viols = _violations(oracle, cvals)
    mults, pens = _multipliers(oracle, info), _penalties(oracle)      # those of the step, before the update
    # the multipliers of the augmented Lagrangian, from the accepted design (before anything of the user can throw,
    # so that a checkpoint written after a failure has them); the optimiser keeps its step size
    end_iteration!(oracle, zk) && objective_changed!(opt, state)
    mvals = try
      map(_monitor, keys(monitors), values(monitors))
    catch e
      e isa InterruptException && rethrow()
      _record!(hist, row, cvals, mults, map(_ -> NaN, values(monitors)), cont_buf, viols, pens)
      fail("a monitor threw at iteration $k: " * first(sprint(showerror, e), 300))
    end
    _record!(hist, row, cvals, mults, mvals, cont_buf, viols, pens)
    _print_row(pr, hist)
    r = _stop_reason(stop, info, hist, run.stage_start, ctx)
    # convergence also needs the constraints met to their tolerances
    (r !== nothing && !_constraints_met(oracle, info, opt, viols)) && (r = nothing)
    done = false
    if r !== nothing
      if schedule === nothing || run.stage == schedule.nstages
        done = true
      elseif schedule.close_early
        run.stage_iters = schedule.every       # end of the stage: jump at the next iteration
      end
    end
    checkpoint === nothing || k % checkpoint.every == 0 &&
      _save_state!(checkpoint, problem, method, opt, state, zk, hist, k, run.stage, run.stage_iters, cont_names, oracle)
    # after the checkpoint: an error of the driver does not lose it
    post !== nothing && _post_due(post, k) && _postprocess!(post, oracle, hist)
    done && return :converged, r
  end
  :maxiter, nothing
end

# a jump whose first design fails: back to the previous stage, so the checkpoint and β match the last accepted design
function _undo_jump!(schedule, run, ctx, cont_buf, opt, state, oracle)
  set_context!(ctx, :continuation, _apply_stage!(schedule, run.stage, cont_buf))
  _invalidate_snapshot!(oracle)
  reset!(opt, state)
  nothing
end

# first finite entry of v[1:n] (NaN if none): the reference of the relative criteria
function _first_finite(v, n)
  for i in 1:n
    isfinite(v[i]) && return v[i]
  end
  NaN
end
