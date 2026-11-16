# The common optimisation loop: continuation, step rejection, stopping, history, checkpoint.

"""
    RejectStep(; shrink = 0.5, max = 5)

What to do when the forward problem does not converge at the design proposed by the optimiser: go back to the
previous design, call `shrink!` on the optimiser with `shrink` (`0 < shrink < 1`: OC multiplies its move limit) and
try again, at most `max` attempts per iteration; then throw [`OptimizationFailure`](@ref). `RejectStep()` is the
default of [`solve!`](@ref) (with the defaults, the step can go down to 1/16 of the first one).

Optimisers that evaluate their trials (SiMPL) handle failures inside their line search, so for them this only
applies if a step is accepted at a design that cannot be evaluated again.
"""
struct RejectStep
  shrink::Float64
  max::Int
end

function RejectStep(; shrink = 0.5, max = 5)
  0 < shrink < 1 || throw(ArgumentError("RejectStep: shrink must be in (0, 1)"))
  max >= 1 || throw(ArgumentError("RejectStep: max must be at least 1"))
  RejectStep(Float64(shrink), Int(max))
end

"""
    StopCriteria(; objective = nothing, window = 10, kkt = nothing, kkt_rel = nothing, change = nothing)

Convergence criteria; each one is off unless given, and the first one met stops the optimisation. With a
[`Continuation`](@ref) they only act in the last stage, unless it was built with `close_early = true` (then a criterion
met in an earlier stage ends that stage).

- `objective`: range of the objective over the last `window` iterations of the current stage, `(max - min)`, relative
  to `max(largest |f| in the window, 1e-3 |f₁|)` (`f₁` the objective of the first iteration of the run, so an
  objective that tends to zero still converges), below this value. A range, not the difference between the ends of the
  window, so an oscillation does not pass for convergence.
- `kkt`: optimality measure of the optimiser below this value (absolute: units of the scaled objective). `kkt_rel`:
  the same, relative to the one of the first iteration of the run (off if that one is not a positive number). Only
  with optimisers that compute it (SiMPL; OC gives an error).
- `change`: `maximum(abs, Δz)` of the step below this value. With a Heaviside projection it is rarely reached.

`kkt`, `kkt_rel` and `change` are not checked at the first iteration of a stage (after a continuation jump the first
step restores feasibility and its measures say little).
"""
struct StopCriteria
  objective::Union{Nothing,Float64}
  window::Int
  kkt::Union{Nothing,Float64}
  kkt_rel::Union{Nothing,Float64}
  change::Union{Nothing,Float64}
end

function StopCriteria(; objective = nothing, window = 10, kkt = nothing, kkt_rel = nothing, change = nothing)
  for (k, v) in ((:objective, objective), (:kkt, kkt), (:kkt_rel, kkt_rel), (:change, change))
    v === nothing || v > 0 || throw(ArgumentError("StopCriteria: $k must be positive"))
  end
  window >= 2 || throw(ArgumentError("StopCriteria: window must be at least 2"))
  f(v) = v === nothing ? nothing : Float64(v)
  StopCriteria(f(objective), Int(window), f(kkt), f(kkt_rel), f(change))
end

"""
    default_stop(optimizer) -> Union{StopCriteria,Nothing}

Stopping criteria of [`solve!`](@ref) when `stop` is not given: `StopCriteria(objective = 1e-3)` (range of the
objective below 0.1 % over the last 10 iterations of the last stage), for every optimiser. With a projection, `kkt`
and `change` are rarely reached; the objective settles.
"""
default_stop(::Optimizer) = StopCriteria(objective = 1e-3)

# `stop` and `on_failure` not given (`stop = nothing` means maxiter only; `on_failure` must be a RejectStep)
struct _DefaultStop end
struct _DefaultFailure end

const HISTORY_COLUMNS = (:iteration, :objective, :constraint, :kkt, :change, :step, :attempts, :forward, :adjoint, :filter,
                         :multiplier, :grey)

"""
    Continuation(; every, close_early = false, name = object => values, …)

Continuation of parameters in stages: `Continuation(β = H => [1, 2, 4, 8, 16, 32]; every = 40)` uses `β = 1` in the
first stage, `2` in the second, and so on. A stage ends after `every` iterations, or when the optimiser finds no
acceptable step; with `close_early = true`, also when the [`StopCriteria`](@ref) are met (by default they only act in
the last stage: early stages can look converged on a grey plateau before the topology forms). The optimisation can
only stop in the last stage. At each jump the memory of the
optimiser is forgotten (`reset!`), which costs one more forward solve. All value lists have the same length (the
number of stages). Supported objects: [`Heaviside`](@ref) (β, through `set_beta!`) and `OC` (its move limit, through
`set_move!`: `Continuation(β = H => [1, 2, 4, 8, 16], move = oc => [0.2, 0.2, 0.1, 0.05, 0.025]; every = 50)`, the
same `oc` given to `SIMP`).
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
  isempty(kwargs) && throw(ArgumentError("Continuation: give at least one parameter, for example β = H => [1, 2, 4]"))
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
  for (o, v) in zip(objects, vals), x in v
    _valid_parameter(o, x) || throw(ArgumentError("Continuation: value $x not valid for $(typeof(o))"))
  end
  n = length(first(vals))
  n >= 1 && all(v -> length(v) == n, vals) ||
    throw(ArgumentError("Continuation: every parameter needs the same number of values (stages)"))
  Continuation(names, objects, vals, Int(every), n, close_early)
end

_set_parameter!(h::Heaviside, v) = set_beta!(h, v)
_valid_parameter(h::Heaviside, v) = isfinite(v) && v > 0
_set_parameter!(o::OC, v) = set_move!(o, v)
_valid_parameter(o::OC, v) = _valid_move(o, v)

function _apply_stage!(c::Continuation, s, buf)
  foreach((o, v) -> _set_parameter!(o, v[s]), c.objects, c.values)
  for (j, v) in enumerate(c.values)
    buf[j] = v[s]
  end
  NamedTuple{c.names}(Tuple(v[s] for v in c.values))
end

"""
    Checkpoint(path; every)

Writes the state of the optimisation to the folder `path` every `every` accepted iterations, and also when the
optimisation stops or fails: `checkpoint.toml` (iteration, continuation stage, scales of the functionals, sizes and
the names of the other two files), `z_<iteration>.bin` (the design, raw `Float64`) and `history_<iteration>.csv`.
The design and the history are written first and `checkpoint.toml` last, under a temporary name and then renamed, so
the three files always belong to the same iteration; older files are then deleted.
Resume with `solve!(…; resume = load_checkpoint(path))`.
"""
struct Checkpoint
  path::String
  every::Int
end

function Checkpoint(path::AbstractString; every::Integer)
  every >= 1 || throw(ArgumentError("Checkpoint: every must be at least 1"))
  Checkpoint(String(path), Int(every))
end

"""
    Grey(θ::PhysicalFields, dΩ; field = nothing)

Monitor of the measure of grey of a physical density, `M_nd = 4/|Ω| ∫ ρ (1 - ρ) dΩ` (Sigmund 2007): 0 for a 0/1
design, 1 for a uniform 0.5. Pass it to `solve!(…; grey = Grey(θ, dΩ))` to add it to the history, the table and the
closing line; it is not a stopping criterion. `field` is mandatory when θ has several fields. Costs one integral per
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
    OptHistory

History of an optimisation, one entry per accepted iteration (vectors): `iteration`, `objective` (scaled value at
the accepted design), `constraint` (scaled value of the constraint, `NaN` without one), `kkt` (optimality
measure, `NaN` if the optimiser does not compute it), `change` (`maximum(abs, Δz)`), `step` (step size or move
limit, the mean over the entries with adaptive limits), `attempts`, `forward`, `adjoint`, `filter` (solves in the iteration), `multiplier` (of the constraint,
as reported by the optimiser), `grey` (measure of grey of the [`Grey`](@ref) monitor, `NaN` without one) and
`continuation` (one column per iteration, one row per continued parameter, names in
`names`).
"""
mutable struct OptHistory
  n::Int
  iteration::Vector{Int}
  objective::Vector{Float64}
  constraint::Vector{Float64}
  kkt::Vector{Float64}
  change::Vector{Float64}
  step::Vector{Float64}
  attempts::Vector{Int}
  forward::Vector{Int}
  adjoint::Vector{Int}
  filter::Vector{Int}
  multiplier::Vector{Float64}
  grey::Vector{Float64}
  names::Vector{Symbol}
  continuation::Matrix{Float64}
end

function OptHistory(cap::Integer, names)
  f() = fill(NaN, cap)
  i() = zeros(Int, cap)
  OptHistory(0, i(), f(), f(), f(), f(), f(), i(), i(), i(), i(), f(), f(), collect(Symbol, names),
             fill(NaN, length(names), cap))
end

Base.length(h::OptHistory) = h.n

function _record!(h::OptHistory, row, cont)
  k = h.n += 1
  h.iteration[k], h.objective[k], h.constraint[k], h.kkt[k], h.change[k], h.step[k], h.attempts[k], h.forward[k],
    h.adjoint[k], h.filter[k], h.multiplier[k], h.grey[k] = row
  for (j, v) in enumerate(cont)
    h.continuation[j, k] = v
  end
  h
end

function _trim!(h::OptHistory)
  n = h.n
  for v in (h.iteration, h.objective, h.constraint, h.kkt, h.change, h.step, h.attempts, h.forward, h.adjoint, h.filter,
            h.multiplier, h.grey)
    resize!(v, n)
  end
  h.continuation = h.continuation[:, 1:n]
  h
end


"""
    OptResult

Result of [`solve!`](@ref): `status` (`:converged`, `:maxiter` or `:stalled`, the last when the optimiser found no
acceptable step in the last continuation stage; in an earlier stage this ends the stage), `reason` (the criterion
met, or `nothing`), `iterations`, `objective` (scaled, at the final design), `history` ([`OptHistory`](@ref)),
`state` (the optimiser state) and `start_scaling` (the factor `t` by which an infeasible starting design was scaled
towards the lower bound, `nothing` if it was not). The final design is in `z`, with the fields and the forward state
in sync with it.
"""
struct OptResult{S}
  status::Symbol
  reason::Union{Nothing,Symbol}
  iterations::Int
  objective::Float64
  history::OptHistory
  state::S
  start_scaling::Union{Nothing,Float64}
end

# ---------------------------------------------------------------------------------------------------------------
# Checkpoint files

function _atomic_write(f, file)
  tmp = file * ".tmp"
  open(f, tmp, "w")
  mv(tmp, file; force = true)
end

# `state === nothing` (a failure): the memory of the optimiser is not saved, a resumed run starts from `init`
function _write_checkpoint(c::Checkpoint, problem, opt, state, zk, hist, k, stage, stage_iters, cont_names)
  mkpath(c.path)
  zfile, hfile, mfile = "z_$k.bin", "history_$k.csv", "memory_$k.bin"
  _atomic_write(io -> write(io, zk), joinpath(c.path, zfile))
  mem = state === nothing ? nothing : memory(opt, state, zk)
  mem === nothing || _atomic_write(io -> write(io, mem), joinpath(c.path, mfile))
  _atomic_write(joinpath(c.path, hfile)) do io
    println(io, join(vcat(String[String(c) for c in HISTORY_COLUMNS], String[String(c) for c in cont_names]), ","))
    for j in 1:hist.n
      row = Any[getfield(hist, f)[j] for f in HISTORY_COLUMNS]
      append!(row, hist.continuation[:, j])
      println(io, join(string.(row), ","))
    end
  end
  meta = Dict{String,Any}(
    "format" => 4,
    "iteration" => k,
    "stage" => stage,
    "stage_iterations" => stage_iters,
    "length" => length(zk),
    "blocks" => String.(collect(problem.z.names)),
    "continuation" => String.(collect(cont_names)),
    "scales" => [F.scale[] for F in (problem.objective, problem.constraints...)],
    "z_file" => zfile,
    "history_file" => hfile,
    "optimizer" => string(nameof(typeof(opt))))
  mem === nothing || (meta["memory_file"] = mfile; meta["memory_length"] = length(mem))
  _atomic_write(io -> TOML.print(io, meta), joinpath(c.path, "checkpoint.toml"))
  for f in readdir(c.path)
    (occursin(r"^(z|memory)_\d+\.bin$", f) || occursin(r"^history_\d+\.csv$", f)) && f ∉ (zfile, hfile, mfile) &&
      rm(joinpath(c.path, f); force = true)
  end
end

"""
    load_checkpoint(path) -> NamedTuple

Reads a checkpoint written by [`Checkpoint`](@ref): `z`, `iteration`, `stage`, `stage_iterations`, `scales`,
`blocks`, `continuation` (names), `history` (named tuple of columns), `optimizer` (its name) and `memory` (the
memory of the optimiser, see [`memory`](@ref HyperShape.Optimizers.memory), or `nothing`). Pass it to
`solve!(…; resume = …)`. OC saves its adaptive move limits and correction shift and resumes exactly; SiMPL does not
save its memory and restarts its step size. A checkpoint written after a failure does not save the memory.
"""
function load_checkpoint(path::AbstractString)
  meta = TOML.parsefile(joinpath(path, "checkpoint.toml"))
  meta["format"] in (1, 2, 3, 4) || throw(ArgumentError("load_checkpoint: unknown format $(meta["format"])"))
  n = meta["length"]
  z = Vector{Float64}(undef, n)
  open(io -> read!(io, z), joinpath(path, meta["z_file"]))
  lines = readlines(joinpath(path, meta["history_file"]))
  header = Symbol.(split(first(lines), ","))
  rows = [parse.(Float64, split(l, ",")) for l in lines[2:end]]
  cols = NamedTuple{Tuple(header)}(Tuple([r[j] for r in rows] for j in eachindex(header)))
  mem = nothing
  if haskey(meta, "memory_file")                # format 4
    mem = Vector{Float64}(undef, meta["memory_length"])
    open(io -> read!(io, mem), joinpath(path, meta["memory_file"]))
  end
  # format 1 (before the multiplier column): the multiplier is unknown
  haskey(cols, :multiplier) || (cols = merge(cols, (multiplier = fill(NaN, length(rows)),)))
  # formats 1 and 2 (before the grey column)
  haskey(cols, :grey) || (cols = merge(cols, (grey = fill(NaN, length(rows)),)))
  (z = z, iteration = Int(meta["iteration"]), stage = Int(meta["stage"]),
   stage_iterations = Int(meta["stage_iterations"]), scales = Float64.(meta["scales"]),
   blocks = Symbol.(meta["blocks"]), continuation = Symbol.(meta["continuation"]), history = cols,
   optimizer = get(meta, "optimizer", ""), memory = mem)
end

function _prefill!(hist::OptHistory, r)
  cols = r.history
  for k in eachindex(cols.iteration)
    row = (Int(cols.iteration[k]), cols.objective[k], cols.constraint[k], cols.kkt[k], cols.change[k], cols.step[k],
           Int(cols.attempts[k]), Int(cols.forward[k]), Int(cols.adjoint[k]), Int(cols.filter[k]), cols.multiplier[k],
           cols.grey[k])
    _record!(hist, row, Tuple(getproperty(cols, n)[k] for n in hist.names))
  end
  hist
end

# ---------------------------------------------------------------------------------------------------------------
# Solve counters

function _counters(problem)
  J = problem.objective
  fwd = J.S === nothing ? () : (J.S,)
  adj = (J.A === nothing || J.A === J.S) ? () : (J.A,)
  filters = Any[]
  for F in (J, problem.constraints...), op in filter_operators(F.θ)
    any(o -> o === op, filters) || push!(filters, op)
  end
  fwd, adj, Tuple(filters)
end

_count(ops) = sum(num_solves, ops; init = 0)

# ---------------------------------------------------------------------------------------------------------------
# The loop

function _stop_reason(stop::StopCriteria, info::StepInfo, hist::OptHistory, stage_start)
  if stop.objective !== nothing && hist.n - stage_start >= stop.window
    w = view(hist.objective, hist.n-stop.window+1:hist.n)
    lo, hi = extrema(w)
    d = max(maximum(abs, w), 1e-3 * abs(hist.objective[1]))   # an objective tending to zero: floor from the first
    (hi - lo) <= stop.objective * d && return :objective
  end
  hist.n - stage_start >= 2 || return nothing   # the first iteration of a stage (restoration after a jump)
  stop.kkt !== nothing && info.kkt < stop.kkt && return :kkt
  if stop.kkt_rel !== nothing
    k1 = hist.kkt[1]
    isfinite(k1) && k1 > 0 && info.kkt < stop.kkt_rel * k1 && return :kkt_rel
  end
  stop.change !== nothing && info.change < stop.change && return :change
  nothing
end
_stop_reason(::Nothing, info, hist, stage_start) = nothing

"""
    solve!(problem::Optimization, method::SIMP; maxiter, on_failure = RejectStep(), stop = default_stop(optimizer),
           schedule = nothing, checkpoint = nothing, post = nothing, resume = nothing, grey = nothing, verbose = 2,
           io = stdout) -> OptResult

Runs the optimisation from the current values of `z`. Mandatory: `maxiter` (total iterations, resumed ones included).
Optional: `on_failure` ([`RejectStep`](@ref), default `RejectStep(shrink = 0.5, max = 5)`), `stop`
([`StopCriteria`](@ref); default [`default_stop`](@ref): range of the objective below 0.1 % over 10 iterations of
the last stage; `stop = nothing` runs `maxiter` iterations), `schedule` ([`Continuation`](@ref)), `checkpoint` ([`Checkpoint`](@ref)), `post` (a function
`post(ctx)` called after each accepted iteration, with the fields and the forward state at the accepted design; the
adjoint state is still the one of the previous design) and
`resume` (the result of [`load_checkpoint`](@ref)), `grey` (a [`Grey`](@ref) monitor: the measure of grey in the
history, the table and the closing line), `verbose` and `io`. An `OC` whose `move` is continued by `schedule` gets
its own `move` back when `solve!` returns or throws.

Screen output (`verbose`, default `2`, printed to `io`): `0` nothing; `1` a banner (variables, constraint, metric,
continuation, stopping criteria and failure handling, each marked when it is a default), the events (continuation
jumps, rejected steps, stalls, a scaled start), a closing
✓/⚠/✗ line and a summary of solves and time; `2` also a row per iteration: objective, its relative change,
constraint (red above 1e-3), optimality measure, change of z, step, multiplier, continued parameters, forward,
adjoint and filter solves, and time. `true` is `2`, `false` is `0`. The summary counts every solve of the call
(those of rejected or stalled trials, the setup and the final synchronisation included).

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
"""
function solve!(problem::Optimization, method::SIMP; maxiter::Integer, on_failure = _DefaultFailure(), stop = _DefaultStop(),
                schedule = nothing, checkpoint = nothing, post = nothing, resume = nothing, grey = nothing, verbose = 2,
                io::IO = stdout)
  z, ctx, opt = problem.z, problem.ctx, method.optimizer
  grey === nothing || grey isa Grey || throw(ArgumentError("solve!: grey must be a Grey monitor, Grey(θ, dΩ)"))
  defaults = (stop = stop isa _DefaultStop, on_failure = on_failure isa _DefaultFailure)
  defaults.stop && (stop = default_stop(opt))
  defaults.on_failure && (on_failure = RejectStep())
  on_failure isa RejectStep || throw(ArgumentError("solve!: on_failure must be a RejectStep"))
  stop === nothing || stop isa StopCriteria || throw(ArgumentError("solve!: stop must be a StopCriteria"))
  schedule === nothing || schedule isa Continuation || throw(ArgumentError("solve!: schedule must be a Continuation"))
  schedule === nothing || all(o -> !(o isa OC) || o === opt, schedule.objects) ||
    throw(ArgumentError("solve!: a Continuation of move must act on the OC given to SIMP"))
  checkpoint === nothing || checkpoint isa Checkpoint || throw(ArgumentError("solve!: checkpoint must be a Checkpoint"))
  maxiter >= 1 || throw(ArgumentError("solve!: maxiter must be at least 1"))
  pr = _Printer(io, verbose, schedule, grey !== nothing)
  # a nonlinear constraint is met to the tolerance of the optimiser: above it, the value is shown in red
  any(c -> !c.linear, problem.constraints) && (pr.red = min(_RED_CONSTRAINT, constraint_tol(opt)))
  lb, ub = get_bounds(z)
  (all(isfinite, lb) && all(isfinite, ub)) || throw(ArgumentError("SIMP: the bounds of z must be finite"))
  cont_names = schedule === nothing ? () : schedule.names
  Fs = (problem.objective, problem.constraints...)

  k0, stage, stage_iters = 0, schedule === nothing ? 0 : 1, 0
  hist = OptHistory(maxiter, cont_names)
  if resume !== nothing
    length(resume.z) == length(z) || throw(ArgumentError("solve!: the checkpoint has $(length(resume.z)) values, z has $(length(z))"))
    resume.blocks == collect(z.names) || throw(ArgumentError("solve!: the checkpoint has blocks $(resume.blocks), z has $(collect(z.names))"))
    resume.continuation == collect(cont_names) ||
      throw(ArgumentError("solve!: the checkpoint continues $(resume.continuation), the schedule $(collect(cont_names))"))
    nst = schedule === nothing ? 0 : schedule.nstages
    (schedule === nothing ? resume.stage == 0 : 1 <= resume.stage <= nst) ||
      throw(ArgumentError("solve!: the checkpoint is at stage $(resume.stage); the schedule has $nst stages"))
    length(resume.scales) == length(Fs) || throw(ArgumentError("solve!: the checkpoint has $(length(resume.scales)) functionals"))
    resume.iteration < maxiter || throw(ArgumentError("solve!: the checkpoint is already at iteration $(resume.iteration) ≥ maxiter"))
    set_values!(z, resume.z)
    # a scale fixed by :initial is restored; a scale given by the user stays
    foreach((F, s) -> (F.scale_mode === :initial && (F.scale[] = s)), Fs, resume.scales)
    k0, stage, stage_iters = resume.iteration, resume.stage, resume.stage_iterations
    _prefill!(hist, resume)
  end

  # an OC whose move limit the schedule continues gets its own value back at the end (the stages change it)
  move0 = opt isa OC ? opt.move : 0.0
  try
  # the parameters of the first stage are applied before the banner, which shows them
  cont_buf = zeros(length(cont_names))
  set_context!(ctx, :continuation, schedule === nothing ? NamedTuple() : _apply_stage!(schedule, stage, cont_buf))
  _print_banner(pr, problem, method, maxiter, stop, on_failure, defaults, schedule, resume, grey)
  set_context!(ctx, :iteration, k0)
  set_context!(ctx, :attempt, 0)
  set_context!(ctx, :trial, 0)

  oracle = FunctionalOracle(problem, method.metric)
  ops = _counters(problem)
  cstart = map(_count, ops)
  _is_linear(oracle) && _setup_constraint!(oracle, only(problem.constraints), ctx)
  set_context!(ctx, :phase, :optimization)
  zk = oracle.zk
  copyto!(zk, get_values(z))
  t_start = nothing
  if resume === nothing                       # a resumed design is the one the run left
    t_start = _make_feasible!(oracle, zk)
    t_start === nothing || _print_event(pr, "· the starting design violated the constraint; it was scaled " *
                                            "towards the lower bound by $(round(t_start, sigdigits = 4)) to meet it", :yellow)
  end
  zprev = similar(zk)
  state = init(opt, zk, oracle)
  if resume !== nothing && get(resume, :memory, nothing) !== nothing
    if resume.optimizer == string(nameof(typeof(opt)))
      restore_memory!(opt, state, zk, resume.memory)
    else
      _print_event(pr, "· the memory of the optimiser was not restored (checkpoint written by $(resume.optimizer))", :yellow)
    end
  end
  fwd, adj, filters = ops
  run = _LoopState(k0, stage, stage_iters, hist.n - stage_iters, cstart, k0)   # what a failure needs
  status, reason = _iterate!(problem, opt, oracle, state, zk, zprev, hist, run, ctx, cont_buf, cont_names, maxiter,
                             on_failure, stop, schedule, checkpoint, post, grey, fwd, adj, filters, pr)

  f_final = _sync!(oracle)                     # fields and forward state at the final design
  _print_end(pr, status, reason, run.k, f_final, grey === nothing ? NaN : grey_value(grey))
  _print_summary(pr, map(_count, ops) .- cstart, run.k - k0)
  checkpoint === nothing ||
    _write_checkpoint(checkpoint, problem, opt, state, zk, hist, run.k, run.stage, run.stage_iters, cont_names)
  OptResult(status, reason, run.k, f_final, _trim!(hist), state, t_start)
  finally
    opt isa OC && set_move!(opt, move0)
  end
end

# A start that violates the constraint is scaled towards the lower bound, z = lb + t (z - lb), so that it holds (with
# equality for a linear constraint, -1e-6 ≤ g ≤ 0 for a nonlinear one): the pattern of the design is kept, the bounds
# hold, and optimisers that need a feasible start (SiMPL) can start.
function _make_feasible!(o, zk)
  o.C === nothing && return nothing
  o.a === nothing && return _make_feasible_nonlinear!(o, zk)
  a, V = o.a, o.V
  az = dot(a, zk)
  az - V <= 1e-12 * max(abs(V), 1.0) && return nothing
  alb = dot(a, o.lb)
  (all(>=(0), a) && alb < V) ||
    throw(ArgumentError("Optimization: the linear constraint cannot be met within the bounds (aᵀlb ≥ V or a has negative entries)"))
  t = (V - alb) / (az - alb)
  @. zk = ifelse(a > 0, o.lb + t * (zk - o.lb), zk)     # entries outside the constraint are left as they are
  set_values!(o.z, zk)
  t
end

# t by regula falsi (Illinois) on g(lb + t (z - lb)) = -tol/2 in (0, 1], on the entries the constraint depends on (a
# nonzero derivative at the start); each evaluation costs the filter solves
function _make_feasible_nonlinear!(o, zk; tol = 1e-6, maxit = 60)
  g1 = constraint(o, zk)
  isfinite(g1) || throw(ArgumentError("Optimization: the constraint is not finite at the starting design"))
  g1 <= 0 && return nothing
  # entries the constraint depends on: nonzero derivative at the start or at the middle of the bounds (a saturated
  # projection gives exact zeros where the constraint still depends on the entry)
  acts = constraint_gradient!(similar(zk), o, zk) .!= 0
  zmid = (o.lb .+ o.ub) ./ 2
  constraint(o, zmid)
  acts .|= constraint_gradient!(similar(zk), o, zmid) .!= 0
  constraint(o, zk)
  z0 = copy(zk)
  zt = similar(zk)
  gt(t) = (@. zt = ifelse(acts, o.lb + t * (z0 - o.lb), z0); constraint(o, zt) + tol / 2)
  t0, t1 = 0.0, 1.0
  h0, h1 = gt(t0), g1 + tol / 2
  h0 < 0 || throw(ArgumentError("Optimization: the constraint cannot be met within the bounds (it is violated at the lower bound)"))
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
  constraint(o, zt) <= 0 || (t = t0; @. zt = ifelse(acts, o.lb + t * (z0 - o.lb), z0))   # end on the feasible side
  copyto!(zk, zt)
  set_values!(o.z, zk)
  t
end

mutable struct _LoopState
  k::Int
  stage::Int
  stage_iters::Int
  stage_start::Int    # history length when the current stage started
  cstart::NTuple{3,Int}   # solve counters at the start of the call
  k0::Int                 # iterations done before the call (resume)
end

function _fail(msg, problem, opt, oracle, zk, hist, run, cont_names, checkpoint, pr, ops)
  synced = try
    isfinite(_sync!(oracle))                   # fields and forward state at the last accepted design
  catch e
    e isa InterruptException && rethrow()
    false
  end
  synced || (msg *= " (the forward state could not be solved again at the last accepted design)")
  _print_failure(pr, msg, run.k)
  _print_summary(pr, map(_count, ops) .- run.cstart, run.k - run.k0)
  checkpoint === nothing ||
    _write_checkpoint(checkpoint, problem, opt, nothing, zk, hist, run.k, run.stage, run.stage_iters, cont_names)
  throw(OptimizationFailure(msg, _trim!(deepcopy(hist)), run.k))
end

function _iterate!(problem, opt, oracle, state, zk, zprev, hist, run, ctx, cont_buf, cont_names, maxiter, on_failure,
                   stop, schedule, checkpoint, post, grey, fwd, adj, filters, pr)
  fail(msg) = _fail(msg, problem, opt, oracle, zk, hist, run, cont_names, checkpoint, pr, (fwd, adj, filters))
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
      isfinite(f_acc) && break
      copyto!(zk, zprev)
      if ctx.attempt >= on_failure.max
        jumped && _undo_jump!(schedule, run, ctx, cont_buf, opt, state, oracle)
        fail("the forward problem did not converge at the $(on_failure.max) designs proposed in iteration $k")
      end
      shrink!(opt, state, on_failure.shrink)
      set_context!(ctx, :attempt, ctx.attempt + 1)
      _print_event(pr, "✗ iteration $k: the forward problem did not converge at the proposed design; step shrunk by " *
                       "$(on_failure.shrink), attempt $(ctx.attempt) of $(on_failure.max)", :red)
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
    if stop !== nothing && (stop.kkt !== nothing || stop.kkt_rel !== nothing) && isnan(info.kkt)
      throw(ArgumentError("StopCriteria: kkt needs an optimiser that computes an optimality measure ($(nameof(typeof(opt))) does not)"))
    end
    run.k = k
    run.stage_iters += 1
    c1 = (_count(fwd), _count(adj), _count(filters))
    _record!(hist, (k, f_acc, _constraint_value(oracle, zk), info.kkt, info.change, info.step, ctx.attempt,
                    c1[1] - c0[1], c1[2] - c0[2], c1[3] - c0[3], info.multiplier,
                    grey === nothing ? NaN : grey_value(grey)), cont_buf)
    _print_row(pr, hist)
    post === nothing || post(ctx)
    r = _stop_reason(stop, info, hist, run.stage_start)
    # with a nonlinear constraint, convergence also needs the constraint met to the tolerance of the optimiser
    (r !== nothing && !_is_linear(oracle) && has_constraint(oracle) && info.violation > constraint_tol(opt)) && (r = nothing)
    done = false
    if r !== nothing
      if schedule === nothing || run.stage == schedule.nstages
        done = true
      elseif schedule.close_early
        run.stage_iters = schedule.every       # end of the stage: jump at the next iteration
      end
    end
    checkpoint === nothing || k % checkpoint.every == 0 &&
      _write_checkpoint(checkpoint, problem, opt, state, zk, hist, k, run.stage, run.stage_iters, cont_names)
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
