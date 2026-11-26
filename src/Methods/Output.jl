# Screen output of the optimisation loop, in the style of HyperFEM's Newton log: a banner, a row per iteration,
# events (continuation jumps, rejected steps, stalls) as indented coloured lines, a closing ✓/⚠/✗ line and a summary.
# Levels: 0 silent; 1 banner, events, closing line and summary; 2 (default) also a row per iteration.
# Colours only on a terminal (`printstyled` writes plain text to files and buffers).

mutable struct _Printer{IO_}
  io::IO_
  level::Int
  t0::Float64
  tlast::Float64
  header::Bool          # the column header is pending: printed before the next row
  widths::Vector{Int}   # widths of the continuation columns
  red::Float64          # a constraint value above this is shown in red
  monitors::Vector{Symbol}   # one column per monitor
  multi::Bool           # several constraints: the largest violation and multiplier
  tols::Vector{Float64} # a constraint violated beyond its tolerance is shown in red
end

_verbose_level(v::Bool) = v ? 2 : 0
_verbose_level(v::Integer) = (0 <= v <= 2 || throw(ArgumentError("solve!: verbose must be 0, 1, 2 or a Bool")); Int(v))
_verbose_level(v) = throw(ArgumentError("solve!: verbose must be 0, 1, 2 or a Bool"))

function _Printer(io, verbose, schedule, monitors = (), multi = false)
  widths = schedule === nothing ? Int[] :
           [max(6, textwidth(string(n)), maximum(x -> textwidth(_g(x)), v)) for (n, v) in zip(schedule.names, schedule.values)]
  _Printer(io, _verbose_level(verbose), time(), time(), false, widths, _RED_CONSTRAINT, collect(Symbol, monitors), multi,
           Float64[])
end

_mwidth(n) = max(7, textwidth(string(n)))

# a constraint value above this (scaled) is shown in red; OC meets the constraint only to its bisection tolerance
const _RED_CONSTRAINT = 1e-3

_styled(p::_Printer, s, color = :normal; bold = false) = printstyled(p.io, s; color, bold)

_e(x) = isfinite(x) ? @sprintf("%9.2e", x) : "        —"

_pad(s, w) = " "^max(0, w - textwidth(s)) * s

function _header(p::_Printer, names)
  cont = join(("  " * _pad(string(n), w) for (n, w) in zip(names, p.widths)))
  @sprintf("  %5s  %13s  %9s  %10s  %9s  %9s  %9s  %9s  %9s", "it", "objective", "Δobj", p.multi ? "max viol" : "constraint",
           "kkt", "S", "change", "step", p.multi ? "max |λ|" : "λ") * cont * join(("  " * _pad(string(n), _mwidth(n)) for n in p.monitors)) *
    @sprintf("  %11s  %8s", "f/a/F", "t[s]")
end

_rule(p::_Printer) = _styled(p, "  " * "─"^(textwidth(_header(p, ntuple(_ -> "", length(p.widths)))) - 2) * "\n", :light_black)

function _describe_stop(stop)
  stop === nothing && return "maxiter only"
  stop isa StopCriteria || return "function of the user"
  parts = String[]
  stop.objective_range !== nothing &&
    push!(parts, "relative objective range < $(stop.objective_range)" *
                 (stop.objective_atol === nothing ? "" : " or < $(stop.objective_atol) absolute") *
                 " over $(stop.window) iterations")
  stop.kkt_tol !== nothing && push!(parts, "kkt < $(stop.kkt_tol)")
  stop.kkt_rtol !== nothing && push!(parts, "kkt/kkt₁ < $(stop.kkt_rtol)")
  stop.stationarity_tol !== nothing && push!(parts, "S < $(stop.stationarity_tol)")
  stop.stationarity_rtol !== nothing && push!(parts, "S/S₁ < $(stop.stationarity_rtol)")
  stop.change_tol !== nothing && push!(parts, "change < $(stop.change_tol)")
  stop.rate_tol !== nothing && push!(parts, "change/step < $(stop.rate_tol)")
  stop.phase_tol !== nothing && push!(parts, "phase change < $(stop.phase_tol) |D| over $(stop.phase_window) iterations")
  stop.material_tol !== nothing && push!(parts, "material change < $(stop.material_tol) over $(stop.material_window) iterations")
  stop.grey_tol !== nothing && push!(parts, "grey < $(stop.grey_tol)")
  isempty(parts) ? "maxiter only" : stop.combine === :all ?
    join(parts, " and ") * ", with the constraints met in the windows" : join(parts, " or ")
end

_active(stop::StopCriteria) = any(x -> x !== nothing, (stop.objective_range, stop.kkt_tol, stop.kkt_rtol, stop.stationarity_tol,
                                         stop.stationarity_rtol, stop.change_tol, stop.rate_tol, stop.phase_tol,
                                         stop.material_tol, stop.grey_tol))
_active(stop) = true                     # a function of the user
_stage_rule(stop, schedule) = (stop === nothing || schedule === nothing || !_active(stop)) ? "" :
                              schedule.close_early ? " · every stage (close_early)" : " · last stage only"

_metric_name(m::Euclidean) = "Euclidean"
_constraint_label(problem, opt, native = 1:length(problem.constraints), tol = NaN) =
  join((string(n) * _bounds_label(c) * ", " * (is_linear(c.F) ? "linear" : "nonlinear") *
        (i in native ? (is_linear(c.F) ? "" : " (tol $(_g_or_e(constraint_tol(opt))))") * (length(native) < length(problem.constraints) ? ", native" : "") :
                       ", augmented Lagrangian (tol $(_g_or_e(tol)))")
        for (i, (n, c)) in enumerate(pairs(problem.constraints))), "; ")
_bounds_label(c) = (c.lo == -Inf && c.hi == 0) ? "" : c.lo == c.hi ? " = $(_sig(c.hi))" :
                   c.lo == -Inf ? " ≤ $(_sig(c.hi))" : c.hi == Inf ? " ≥ $(_sig(c.lo))" :
                   " in [$(_sig(c.lo)), $(_sig(c.hi))]"
_sig(x) = string(round(x; sigdigits = 4))
_g_or_e(x) = isfinite(x) ? @sprintf("%.0e", x) : "—"
metric_label(method::DensityMethod) = _metric_name(method.metric) * (method.default_metric ? " (default)" : "") *
                                      (method.optimizer isa OC ? " (OC: only for the stationarity measure)" : "")
_metric_name(m::LumpedMass) = "lumped mass"
_metric_name(m::LumpedMass{Nothing}) = "lumped mass of the design space"
_metric_name(m::AbstractVector) = "weights of the user"
_metric_name(m) = string(nameof(typeof(m)))
_metric_name(m::NamedTuple) = join(("$k: $(_metric_name(v))" for (k, v) in pairs(m)), ", ")

function _print_banner(p::_Printer, problem, method, maxiter, stop, on_failure, defaults, schedule, resume,
                       native = 1:length(problem.constraints), discretization = FixedMesh())
  p.level >= 1 || return
  opt = stepper(method)
  z = problem.z
  title = " HyperShape · $(method_label(method)) "
  W = textwidth(_header(p, ntuple(_ -> "", length(p.widths))))   # as wide as the table
  _styled(p, "━━" * title * "━"^max(3, W - 2 - textwidth(title)) * "\n", :cyan; bold = true)
  blocks = join(("$n ($(length(r))" * (b.zone === nothing ? "" : ", $(b.zone.ncells) frozen cells") * ")"
                 for (n, r, b) in zip(z.names, z.ranges, z.blocks)), ", ")
  nc = length(problem.constraints)
  lag = augmented_lagrangian(method)
  ml = metric_label(method)
  println(p.io, "  design variables $(length(z)): $blocks · constraint", nc > 1 ? "s" : "", ": ",
          nc == 0 ? "none" : _constraint_label(problem, opt, native, lag.constraint_tol), ml === nothing ? "" : " · metric: " * ml)
  if length(native) < nc
    println(p.io, "  augmented Lagrangian: ", lag.update, (method isa DensityMethod && method.default_lagrangian) ? " (default)" : "",
            ", multipliers updated every iteration")
    opt isa OC && _styled(p, "  OC is heuristic for an augmented Lagrangian (an objective of mixed sign): SiMPL is the safe choice\n", :yellow)
  end
  if schedule !== nothing
    for (n, v) in zip(schedule.names, schedule.values)
      println(p.io, "  continuation $n: ", join((_g(x) for x in v), " → "), " (every $(schedule.every) iterations)")
    end
  end
  println(p.io, "  stop: ", _describe_stop(stop), defaults.stop ? " (default)" : "", _stage_rule(stop, schedule),
          " · maxiter $maxiter", resume === nothing ? "" : " · resumed at iteration $(resume.iteration)")
  println(p.io, "  on failure: reject step, shrink $(_g(on_failure.shrink)), at most $(on_failure.max_attempts) attempts",
          defaults.on_failure ? " (default)" : "")
  println(p.io, "  discretization: ", discretization_label(discretization),
          get(defaults, :discretization, false) ? " (default)" : "",
          problem.setup === nothing ? "" : " · problem from design_setup/physics_setup")
  for (line, color) in banner_lines(method, problem)
    _styled(p, "  " * line * "\n", color)
  end
  _print_columns(p, schedule)
end

_g(x) = x == round(x) && abs(x) < 1e6 ? string(Int(x)) : @sprintf("%.3g", x)

_print_columns(p::_Printer, schedule) = (p.level >= 2 && (p.header = true); nothing)

function _flush_columns(p::_Printer, names)
  p.header || return
  p.header = false
  _rule(p)
  _styled(p, _header(p, names) * "\n"; bold = true)
  _rule(p)
end

function _print_row(p::_Printer, hist::OptHistory)
  p.level >= 2 || return nothing
  _flush_columns(p, hist.continuation_names)
  k = hist.n
  t = time()
  dt = t - p.tlast
  p.tlast = t
  f = hist.objective[k]
  fp = k > 1 ? hist.objective[k-1] : NaN
  df = fp != 0 ? (f - fp) / abs(fp) : NaN
  if p.multi
    g = maximum(view(hist.violation_values, :, k))
    μ = maximum(x -> isfinite(x) ? abs(x) : -Inf, view(hist.multiplier_values, :, k))
  else
    g = size(hist.constraint_values, 1) == 0 ? NaN : hist.constraint_values[1, k]
    μ = size(hist.multiplier_values, 1) == 0 ? NaN : hist.multiplier_values[1, k]
  end
  @printf(p.io, "  %5d  %13.6e  %9s  ", hist.iteration[k], f, isfinite(df) ? @sprintf("%+9.1e", df) : "        —")
  # red: a constraint violated beyond its tolerance (its violation, so a lower bound that holds is not red)
  bad = isempty(p.tols) ? g > p.red :
        any(j -> hist.violation_values[j, k] > p.tols[j], eachindex(p.tols))
  if isfinite(g)
    _styled(p, @sprintf("%10.2e", g), bad ? :red : :normal)
  else
    print(p.io, "         —")
  end
  @printf(p.io, "  %9s  %9s  %9s  %9s  %9s", _e(hist.kkt[k]), _e(hist.stationarity[k]), _e(hist.change[k]),
          _e(hist.step[k]), _e(μ))
  for j in axes(hist.continuation_values, 1)
    print(p.io, "  ", _pad(_g(hist.continuation_values[j, k]), p.widths[j]))
  end
  for (j, n) in enumerate(p.monitors)
    v = hist.monitor_values[j, k]
    print(p.io, "  ", _pad(isfinite(v) ? @sprintf("%.4g", v) : "—", _mwidth(n)))
  end
  @printf(p.io, "  %11s  %8.3f", "$(hist.forward[k])/$(hist.adjoint[k])/$(hist.filter[k])", dt)
  hist.attempts[k] > 1 && _styled(p, "  ($(hist.attempts[k]) attempts)", :yellow)
  println(p.io)
  nothing
end

function _print_event(p::_Printer, msg, color)
  p.level >= 1 || return
  _styled(p, "        " * msg * "\n", color)
end

function _print_jump(p::_Printer, schedule, stage)
  p.level >= 1 || return
  vals = join(("$n = $(_g(v[stage]))" for (n, v) in zip(schedule.names, schedule.values)), ", ")
  _print_event(p, "→ $vals (stage $stage of $(schedule.nstages))", :cyan)
  _print_columns(p, schedule)
end

const _STOP_TEXT = Dict(:change_tol => "change of z below the tolerance",
                        :rate_tol => "rate of change of z (change/step) below the tolerance",
                        :phase_tol => "fraction of the domain that changes phase below the tolerance over the window",
                        :material_tol => "change of the material below the tolerance over the window",
                        :grey_tol => "measure of grey of the design below the tolerance",
                        :all => "all the stopping criteria met", :kkt_tol => "optimality measure below the tolerance",
                        :kkt_rtol => "optimality measure below the tolerance relative to the first iteration",
                        :stationarity_tol => "stationarity measure below the tolerance",
                        :stationarity_rtol => "stationarity measure below the tolerance relative to the first iteration",
                        :objective_range => "relative range of the objective below the tolerance over the window",
                        :custom => "criterion of the user met")

function _print_end(p::_Printer, status, reason, k, f, monitors = NamedTuple())
  p.level >= 1 || return
  p.level >= 2 && (p.header = false; _rule(p))
  if status === :converged
    _styled(p, "  ✓ ", :green; bold = true)
    print(p.io, "converged: ", get(_STOP_TEXT, reason, string(reason)), " after $k iterations, objective ",
          @sprintf("%.6e", f))
  elseif status === :maxiter
    _styled(p, "  ⚠ ", :yellow; bold = true)
    print(p.io, "maximum number of iterations ($k), objective ", @sprintf("%.6e", f))
  else
    _styled(p, "  ⚠ ", :yellow; bold = true)
    print(p.io, "stalled: no acceptable step after $k iterations, objective ", @sprintf("%.6e", f))
  end
  for (n, v) in pairs(monitors)
    isfinite(v) && print(p.io, ", $n ", @sprintf("%.4g", v))
  end
  println(p.io)
end

function _print_failure(p::_Printer, msg, k)
  p.level >= 1 || return
  p.level >= 2 && (p.header = false; _rule(p))
  _styled(p, "  ✗ ", :red; bold = true)
  println(p.io, "failed after $k accepted iterations: ", msg)
end

# solves: (forward, adjoint, filter) solved in this call, the setup and the final synchronisation included; `note`: a
# line of the step (ReactionDiffusion: its solves and factorisations)
function _print_summary(p::_Printer, solves, iterations, events = 0, note = "")
  p.level >= 1 || return
  t = time() - p.t0
  println(p.io, @sprintf("  solves: forward %d, adjoint %d, filter %d · time %.2f s (%.3f s per iteration)",
                         solves..., t, t / max(iterations, 1)),
          events == 0 ? "" : " · $events rebuild event" * (events == 1 ? "" : "s"))
  isempty(note) || println(p.io, note)
end

# the move limit of OC
banner_lines(m::DensityMethod, problem) =
  m.optimizer isa OC ? [("OC: move $(_g(m.optimizer.move)) · " * (m.optimizer.adapt === nothing ? "fixed (top88)" :
                         "adaptive per entry ×$(_g(m.optimizer.adapt[1]))/×$(_g(m.optimizer.adapt[2])), floor $(_g(m.optimizer.move_min))"), :normal)] :
                       Tuple{String,Symbol}[]
