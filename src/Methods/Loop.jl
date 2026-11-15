# The common optimisation loop: continuation, step rejection, stopping, history, checkpoint.

"""
    RejectStep(; shrink, max)

What to do when the forward problem does not converge at the design proposed by the optimiser: go back to the
previous design, call `shrink!` on the optimiser with `shrink` (`0 < shrink < 1`: OC multiplies its move limit) and
try again, at most `max` attempts per iteration; then throw [`OptimizationFailure`](@ref). Both are mandatory.

Optimisers that evaluate their trials (SiMPL) handle failures inside their line search, so for them this only
applies if a step is accepted at a design that cannot be evaluated again.
"""
struct RejectStep
  shrink::Float64
  max::Int
end

function RejectStep(; shrink, max)
  0 < shrink < 1 || throw(ArgumentError("RejectStep: shrink must be in (0, 1)"))
  max >= 1 || throw(ArgumentError("RejectStep: max must be at least 1"))
  RejectStep(Float64(shrink), Int(max))
end

"""
    StopCriteria(; change = nothing, kkt = nothing, objective = nothing, window = nothing)

Convergence criteria; each one is off unless given, and the first one met stops the optimisation (in the last stage of
the continuation). `change`: `maximum(abs, Δz)` of the step below this value. `kkt`: optimality measure of the
optimiser below this value (SiMPL; an optimiser that does not compute it, such as OC, gives an error). `objective`:
relative change of the objective over the last `window` iterations (mandatory with `objective`) below this value,
counting only iterations of the current continuation stage.
"""
struct StopCriteria
  change::Union{Nothing,Float64}
  kkt::Union{Nothing,Float64}
  objective::Union{Nothing,Float64}
  window::Int
end

function StopCriteria(; change = nothing, kkt = nothing, objective = nothing, window = nothing)
  for (k, v) in ((:change, change), (:kkt, kkt), (:objective, objective))
    v === nothing || v > 0 || throw(ArgumentError("StopCriteria: $k must be positive"))
  end
  objective === nothing || window !== nothing || throw(ArgumentError("StopCriteria: objective needs a window"))
  window = window === nothing ? 1 : window
  window >= 1 || throw(ArgumentError("StopCriteria: window must be at least 1"))
  StopCriteria(change === nothing ? nothing : Float64(change), kkt === nothing ? nothing : Float64(kkt),
               objective === nothing ? nothing : Float64(objective), Int(window))
end

const HISTORY_COLUMNS = (:iteration, :objective, :constraint, :kkt, :change, :step, :attempts, :forward, :adjoint, :filter,
                         :multiplier)

"""
    Continuation(; every, name = object => values, …)

Continuation of parameters in stages: `Continuation(β = H => [1, 2, 4, 8, 16, 32]; every = 40)` uses `β = 1` in the
first stage, `2` in the second, and so on. A stage ends after `every` iterations, or earlier when the
[`StopCriteria`](@ref) are met; the optimisation can only stop in the last stage. At each jump the memory of the
optimiser is forgotten (`reset!`), which costs one more forward solve. All value lists have the same length (the
number of stages). Supported objects: [`Heaviside`](@ref) (β, through `set_beta!`).
"""
struct Continuation{N,O<:Tuple,V<:Tuple}
  names::NTuple{N,Symbol}
  objects::O
  values::V
  every::Int
  nstages::Int
end

function Continuation(; every::Integer, kwargs...)
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
  Continuation(names, objects, vals, Int(every), n)
end

_set_parameter!(h::Heaviside, v) = set_beta!(h, v)
_valid_parameter(h::Heaviside, v) = isfinite(v) && v > 0

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
    OptHistory

History of an optimisation, one entry per accepted iteration (vectors): `iteration`, `objective` (scaled value at
the accepted design), `constraint` (scaled value of the linear constraint, `NaN` without one), `kkt` (optimality
measure, `NaN` if the optimiser does not compute it), `change` (`maximum(abs, Δz)`), `step` (step size or move
limit), `attempts`, `forward`, `adjoint`, `filter` (solves in the iteration), `multiplier` (of the linear constraint,
as reported by the optimiser) and `continuation` (one column per iteration, one row per continued parameter, names in
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
  names::Vector{Symbol}
  continuation::Matrix{Float64}
end

function OptHistory(cap::Integer, names)
  f() = fill(NaN, cap)
  i() = zeros(Int, cap)
  OptHistory(0, i(), f(), f(), f(), f(), f(), i(), i(), i(), i(), f(), collect(Symbol, names), fill(NaN, length(names), cap))
end

Base.length(h::OptHistory) = h.n

function _record!(h::OptHistory, row, cont)
  k = h.n += 1
  h.iteration[k], h.objective[k], h.constraint[k], h.kkt[k], h.change[k], h.step[k], h.attempts[k], h.forward[k],
    h.adjoint[k], h.filter[k], h.multiplier[k] = row
  for (j, v) in enumerate(cont)
    h.continuation[j, k] = v
  end
  h
end

function _trim!(h::OptHistory)
  n = h.n
  for v in (h.iteration, h.objective, h.constraint, h.kkt, h.change, h.step, h.attempts, h.forward, h.adjoint, h.filter,
            h.multiplier)
    resize!(v, n)
  end
  h.continuation = h.continuation[:, 1:n]
  h
end


"""
    OptResult

Result of [`solve!`](@ref): `status` (`:converged`, `:maxiter` or `:stalled`, the last when the optimiser found no
acceptable step in the last continuation stage; in an earlier stage this ends the stage), `reason` (the criterion met, or `nothing`), `iterations`, `objective` (scaled, at the final design),
`history` ([`OptHistory`](@ref)) and `state` (the optimiser state). The final design is in `z`, with the fields and
the forward state in sync with it.
"""
struct OptResult{S}
  status::Symbol
  reason::Union{Nothing,Symbol}
  iterations::Int
  objective::Float64
  history::OptHistory
  state::S
end

# ---------------------------------------------------------------------------------------------------------------
# Checkpoint files

function _atomic_write(f, file)
  tmp = file * ".tmp"
  open(f, tmp, "w")
  mv(tmp, file; force = true)
end

function _write_checkpoint(c::Checkpoint, problem, opt, zk, hist, k, stage, stage_iters, cont_names)
  mkpath(c.path)
  zfile, hfile = "z_$k.bin", "history_$k.csv"
  _atomic_write(io -> write(io, zk), joinpath(c.path, zfile))
  _atomic_write(joinpath(c.path, hfile)) do io
    println(io, join(vcat(String[String(c) for c in HISTORY_COLUMNS], String[String(c) for c in cont_names]), ","))
    for j in 1:hist.n
      row = Any[getfield(hist, f)[j] for f in HISTORY_COLUMNS]
      append!(row, hist.continuation[:, j])
      println(io, join(string.(row), ","))
    end
  end
  meta = Dict{String,Any}(
    "format" => 1,
    "iteration" => k,
    "stage" => stage,
    "stage_iterations" => stage_iters,
    "length" => length(zk),
    "blocks" => String.(collect(problem.z.names)),
    "continuation" => String.(collect(cont_names)),
    "scales" => [F.scale[] for F in (problem.objective, problem.linear_constraints...)],
    "z_file" => zfile,
    "history_file" => hfile,
    "optimizer" => string(nameof(typeof(opt))))
  _atomic_write(io -> TOML.print(io, meta), joinpath(c.path, "checkpoint.toml"))
  for f in readdir(c.path)
    (occursin(r"^z_\d+\.bin$", f) || occursin(r"^history_\d+\.csv$", f)) && f != zfile && f != hfile &&
      rm(joinpath(c.path, f); force = true)
  end
end

"""
    load_checkpoint(path) -> NamedTuple

Reads a checkpoint written by [`Checkpoint`](@ref): `z`, `iteration`, `stage`, `stage_iterations`, `scales`,
`blocks`, `continuation` (names) and `history` (named tuple of columns). Pass it to `solve!(…; resume = …)`. The
memory of the optimiser is not saved: OC resumes exactly, SiMPL restarts its step size.
"""
function load_checkpoint(path::AbstractString)
  meta = TOML.parsefile(joinpath(path, "checkpoint.toml"))
  meta["format"] == 1 || throw(ArgumentError("load_checkpoint: unknown format $(meta["format"])"))
  n = meta["length"]
  z = Vector{Float64}(undef, n)
  open(io -> read!(io, z), joinpath(path, meta["z_file"]))
  lines = readlines(joinpath(path, meta["history_file"]))
  header = Symbol.(split(first(lines), ","))
  rows = [parse.(Float64, split(l, ",")) for l in lines[2:end]]
  cols = NamedTuple{Tuple(header)}(Tuple([r[j] for r in rows] for j in eachindex(header)))
  (z = z, iteration = Int(meta["iteration"]), stage = Int(meta["stage"]),
   stage_iterations = Int(meta["stage_iterations"]), scales = Float64.(meta["scales"]),
   blocks = Symbol.(meta["blocks"]), continuation = Symbol.(meta["continuation"]), history = cols)
end

function _prefill!(hist::OptHistory, r)
  cols = r.history
  for k in eachindex(cols.iteration)
    row = (Int(cols.iteration[k]), cols.objective[k], cols.constraint[k], cols.kkt[k], cols.change[k], cols.step[k],
           Int(cols.attempts[k]), Int(cols.forward[k]), Int(cols.adjoint[k]), Int(cols.filter[k]), cols.multiplier[k])
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
  for F in (J, problem.linear_constraints...), op in filter_operators(F.θ)
    any(o -> o === op, filters) || push!(filters, op)
  end
  fwd, adj, Tuple(filters)
end

_count(ops) = sum(num_solves, ops; init = 0)

# ---------------------------------------------------------------------------------------------------------------
# The loop

function _stop_reason(stop::StopCriteria, info::StepInfo, hist::OptHistory, stage_start)
  stop.change !== nothing && info.change < stop.change && return :change
  stop.kkt !== nothing && info.kkt < stop.kkt && return :kkt
  if stop.objective !== nothing && hist.n - stop.window > stage_start
    f, f0 = hist.objective[hist.n], hist.objective[hist.n-stop.window]
    abs(f - f0) <= stop.objective * abs(f) && return :objective
  end
  nothing
end
_stop_reason(::Nothing, info, hist, stage_start) = nothing

"""
    solve!(problem::Optimization, method::SIMP; maxiter, on_failure, stop = nothing, schedule = nothing,
           checkpoint = nothing, post = nothing, resume = nothing) -> OptResult

Runs the optimisation from the current values of `z`. Mandatory: `maxiter` (total iterations, resumed ones included)
and `on_failure` ([`RejectStep`](@ref)). Optional: `stop` ([`StopCriteria`](@ref); without it the loop runs
`maxiter` iterations), `schedule` ([`Continuation`](@ref)), `checkpoint` ([`Checkpoint`](@ref)), `post` (a function
`post(ctx)` called after each accepted iteration, with the fields and the forward state at the accepted design; the
adjoint state is still the one of the previous design) and
`resume` (the result of [`load_checkpoint`](@ref)), `verbose` and `io`.

Screen output (`verbose`, default `2`, printed to `io`): `0` nothing; `1` a banner (variables, constraint, metric,
continuation, stopping criteria), the events (continuation jumps, rejected steps, stalls, a scaled start), a closing
✓/⚠/✗ line and a summary of solves and time; `2` also a row per iteration: objective, its relative change,
constraint (red if violated), optimality measure, change of z, step, multiplier, continued parameters, forward,
adjoint and filter solves, and time. `true` is `2`, `false` is `0`.

If the starting design (not a resumed one) violates the linear constraint, it is scaled towards the lower bound,
`z = lb + t (z − lb)`, so that the constraint holds with equality (logged with `@info`).

Each iteration: the optimiser takes a step from the current design; the loop evaluates the new design (no solve when
the optimiser already did, as SiMPL); if the forward problem does not converge there, the design is restored, the
step shrunk and tried again. Solves per iteration: OC one filter, one forward, one adjoint (none for a self-adjoint
functional) and one transposed filter; SiMPL one filter and one forward per trial of its line search (usually one),
one adjoint and one transposed filter; one more forward after each continuation jump. The linear constraint costs
nothing in the loop (its vector is computed once at the start).
"""
function solve!(problem::Optimization, method::SIMP; maxiter::Integer, on_failure::RejectStep, stop = nothing,
                schedule = nothing, checkpoint = nothing, post = nothing, resume = nothing, verbose = 2, io::IO = stdout)
  z, ctx, opt = problem.z, problem.ctx, method.optimizer
  pr = _Printer(io, verbose)
  stop === nothing || stop isa StopCriteria || throw(ArgumentError("solve!: stop must be a StopCriteria"))
  schedule === nothing || schedule isa Continuation || throw(ArgumentError("solve!: schedule must be a Continuation"))
  checkpoint === nothing || checkpoint isa Checkpoint || throw(ArgumentError("solve!: checkpoint must be a Checkpoint"))
  maxiter >= 1 || throw(ArgumentError("solve!: maxiter must be at least 1"))
  lb, ub = get_bounds(z)
  (all(isfinite, lb) && all(isfinite, ub)) || throw(ArgumentError("SIMP: the bounds of z must be finite"))
  cont_names = schedule === nothing ? () : schedule.names
  Fs = (problem.objective, problem.linear_constraints...)

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

  _print_banner(pr, problem, method, maxiter, stop, schedule, resume)
  set_context!(ctx, :iteration, k0)
  set_context!(ctx, :attempt, 0)
  set_context!(ctx, :trial, 0)
  cont_buf = zeros(length(cont_names))
  set_context!(ctx, :continuation, schedule === nothing ? NamedTuple() : _apply_stage!(schedule, stage, cont_buf))

  oracle = FunctionalOracle(problem, method.metric)
  isempty(problem.linear_constraints) || _setup_constraint!(oracle, only(problem.linear_constraints), ctx)
  set_context!(ctx, :phase, :optimization)
  zk = oracle.zk
  copyto!(zk, get_values(z))
  if resume === nothing                       # a resumed design is the one the run left
    t = _make_feasible!(oracle, zk)
    if t !== nothing
      msg = "the starting design violated the linear constraint; it was scaled towards the lower bound by " *
            "$(round(t, sigdigits = 4)) to meet it"
      pr.level >= 1 ? _print_event(pr, "· " * msg, :yellow) : @info("solve!: " * msg)
    end
  end
  zprev = similar(zk)
  state = init(opt, zk, oracle)
  fwd, adj, filters = _counters(problem)
  run = _LoopState(k0, stage, stage_iters, hist.n - stage_iters)   # what a failure needs to write the checkpoint
  status, reason = _iterate!(problem, opt, oracle, state, zk, zprev, hist, run, ctx, cont_buf, cont_names, maxiter,
                             on_failure, stop, schedule, checkpoint, post, fwd, adj, filters, pr)

  f_final = _sync!(oracle)                     # fields and forward state at the final design
  _print_end(pr, status, reason, run.k, f_final)
  _print_summary(pr, hist, k0)
  checkpoint === nothing ||
    _write_checkpoint(checkpoint, problem, opt, zk, hist, run.k, run.stage, run.stage_iters, cont_names)
  OptResult(status, reason, run.k, f_final, _trim!(hist), state)
end

# A start that violates the linear constraint is scaled towards the lower bound, z = lb + t (z - lb), so that aᵀz = V:
# the pattern of the design is kept, the bounds hold, and optimisers that need a feasible start (SiMPL) can start.
function _make_feasible!(o::FunctionalOracle, zk)
  o.a === nothing && return nothing
  a, V = o.a, o.V
  az = dot(a, zk)
  az - V <= 1e-12 * max(abs(V), 1.0) && return nothing
  alb = dot(a, o.lb)
  (all(>=(0), a) && alb < V) ||
    throw(ArgumentError("Optimization: the linear constraint cannot be met within the bounds (aᵀlb ≥ V or a has negative entries)"))
  t = (V - alb) / (az - alb)
  @. zk = o.lb + t * (zk - o.lb)
  set_values!(o.z, zk)
  t
end

mutable struct _LoopState
  k::Int
  stage::Int
  stage_iters::Int
  stage_start::Int    # history length when the current stage started
end

function _fail(msg, problem, opt, oracle, zk, hist, run, cont_names, checkpoint, pr)
  synced = try
    isfinite(_sync!(oracle))                   # fields and forward state at the last accepted design
  catch e
    e isa InterruptException && rethrow()
    false
  end
  synced || (msg *= " (the forward state could not be solved again at the last accepted design)")
  _print_failure(pr, msg, run.k)
  checkpoint === nothing ||
    _write_checkpoint(checkpoint, problem, opt, zk, hist, run.k, run.stage, run.stage_iters, cont_names)
  throw(OptimizationFailure(msg, _trim!(deepcopy(hist)), run.k))
end

function _iterate!(problem, opt, oracle, state, zk, zprev, hist, run, ctx, cont_buf, cont_names, maxiter, on_failure,
                   stop, schedule, checkpoint, post, fwd, adj, filters, pr)
  fail(msg) = _fail(msg, problem, opt, oracle, zk, hist, run, cont_names, checkpoint, pr)
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
      run.stage_iters = schedule.every
      continue
    end
    if jumped
      run.stage += 1
      run.stage_iters = 0
      run.stage_start = hist.n
    end
    if stop !== nothing && stop.kkt !== nothing && isnan(info.kkt)
      throw(ArgumentError("StopCriteria: kkt needs an optimiser that computes an optimality measure ($(nameof(typeof(opt))) does not)"))
    end
    run.k = k
    run.stage_iters += 1
    c1 = (_count(fwd), _count(adj), _count(filters))
    _record!(hist, (k, f_acc, _constraint_value(oracle, zk), info.kkt, info.change, info.step, ctx.attempt,
                    c1[1] - c0[1], c1[2] - c0[2], c1[3] - c0[3], info.multiplier), cont_buf)
    _print_row(pr, hist)
    post === nothing || post(ctx)
    r = _stop_reason(stop, info, hist, run.stage_start)
    done = false
    if r !== nothing
      if schedule === nothing || run.stage == schedule.nstages
        done = true
      else
        run.stage_iters = schedule.every       # end of the stage: jump at the next iteration
      end
    end
    checkpoint === nothing || k % checkpoint.every == 0 &&
      _write_checkpoint(checkpoint, problem, opt, zk, hist, k, run.stage, run.stage_iters, cont_names)
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
