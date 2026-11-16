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
end

_verbose_level(v::Bool) = v ? 2 : 0
_verbose_level(v::Integer) = (0 <= v <= 2 || throw(ArgumentError("solve!: verbose must be 0, 1, 2 or a Bool")); Int(v))
_verbose_level(v) = throw(ArgumentError("solve!: verbose must be 0, 1, 2 or a Bool"))

function _Printer(io, verbose, schedule)
  widths = schedule === nothing ? Int[] :
           [max(6, textwidth(string(n)), maximum(x -> textwidth(_g(x)), v)) for (n, v) in zip(schedule.names, schedule.values)]
  _Printer(io, _verbose_level(verbose), time(), time(), false, widths)
end

# a constraint value above this (scaled) is shown in red; OC meets the constraint only to its bisection tolerance
const _RED_CONSTRAINT = 1e-3

_styled(p::_Printer, s, color = :normal; bold = false) = printstyled(p.io, s; color, bold)

_e(x) = isfinite(x) ? @sprintf("%9.2e", x) : "        —"

_pad(s, w) = " "^max(0, w - textwidth(s)) * s

function _header(p::_Printer, names)
  cont = join(("  " * _pad(string(n), w) for (n, w) in zip(names, p.widths)))
  @sprintf("  %5s  %13s  %9s  %10s  %9s  %9s  %9s  %9s", "it", "objective", "Δobj", "constraint", "kkt", "change",
           "step", "μ") * cont * @sprintf("  %11s  %8s", "f/a/F", "t[s]")
end

_rule(p::_Printer) = _styled(p, "  " * "─"^(textwidth(_header(p, ntuple(_ -> "", length(p.widths)))) - 2) * "\n", :light_black)

function _describe_stop(stop)
  stop === nothing && return "maxiter only"
  parts = String[]
  stop.change !== nothing && push!(parts, "change < $(stop.change)")
  stop.kkt !== nothing && push!(parts, "kkt < $(stop.kkt)")
  stop.objective !== nothing && push!(parts, "Δobjective < $(stop.objective) over $(stop.window)")
  isempty(parts) ? "maxiter only" : join(parts, ", ")
end

_metric_name(m::Euclidean) = "Euclidean"
_constraint_label(problem, opt) = only(problem.constraints).linear ? "linear" :
                                  "nonlinear (tol $(_g_or_e(constraint_tol(opt))))"
_g_or_e(x) = isfinite(x) ? @sprintf("%.0e", x) : "—"
_metric_label(method) = method.optimizer isa OC ? "not used by OC" :
                        _metric_name(method.metric) * (method.default_metric ? " (default)" : "")
_metric_name(m::LumpedMass) = "lumped mass"
_metric_name(m::NamedTuple) = join(("$k: $(_metric_name(v))" for (k, v) in pairs(m)), ", ")

function _print_banner(p::_Printer, problem, method, maxiter, stop, on_failure, defaults, schedule, resume)
  p.level >= 1 || return
  opt = method.optimizer
  z = problem.z
  title = " HyperShape · SIMP($(nameof(typeof(opt)))) "
  W = textwidth(_header(p, ntuple(_ -> "", length(p.widths))))   # as wide as the table
  _styled(p, "━━" * title * "━"^max(3, W - 2 - textwidth(title)) * "\n", :cyan; bold = true)
  blocks = join(("$n ($(length(r)))" for (n, r) in zip(z.names, z.ranges)), ", ")
  nc = length(problem.constraints)
  println(p.io, "  design variables $(length(z)): $blocks · constraint: ", nc == 0 ? "none" : _constraint_label(problem, opt), " · metric: ",
          _metric_label(method))
  if schedule !== nothing
    for (n, v) in zip(schedule.names, schedule.values)
      println(p.io, "  continuation $n: ", join((_g(x) for x in v), " → "), " (every $(schedule.every) iterations)")
    end
  end
  println(p.io, "  stop: ", _describe_stop(stop), defaults.stop ? " (default for $(nameof(typeof(opt))))" : "",
          " · maxiter $maxiter", resume === nothing ? "" : " · resumed at iteration $(resume.iteration)")
  println(p.io, "  on failure: reject step, shrink $(_g(on_failure.shrink)), at most $(on_failure.max) attempts",
          defaults.on_failure ? " (default)" : "")
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
  _flush_columns(p, hist.names)
  k = hist.n
  t = time()
  dt = t - p.tlast
  p.tlast = t
  f = hist.objective[k]
  fp = k > 1 ? hist.objective[k-1] : NaN
  df = fp != 0 ? (f - fp) / abs(fp) : NaN
  g = hist.constraint[k]
  @printf(p.io, "  %5d  %13.6e  %9s  ", hist.iteration[k], f, isfinite(df) ? @sprintf("%+9.1e", df) : "        —")
  if isfinite(g)
    _styled(p, @sprintf("%10.2e", g), g > _RED_CONSTRAINT ? :red : :normal)
  else
    print(p.io, "         —")
  end
  @printf(p.io, "  %9s  %9s  %9s  %9s", _e(hist.kkt[k]), _e(hist.change[k]), _e(hist.step[k]), _e(hist.multiplier[k]))
  for j in axes(hist.continuation, 1)
    print(p.io, "  ", _pad(_g(hist.continuation[j, k]), p.widths[j]))
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

const _STOP_TEXT = Dict(:change => "change of z below the tolerance", :kkt => "optimality measure below the tolerance",
                        :objective => "relative change of the objective below the tolerance")

function _print_end(p::_Printer, status, reason, k, f)
  p.level >= 1 || return
  p.level >= 2 && (p.header = false; _rule(p))
  if status === :converged
    _styled(p, "  ✓ ", :green; bold = true)
    println(p.io, "converged: ", get(_STOP_TEXT, reason, string(reason)), " after $k iterations, objective ",
            @sprintf("%.6e", f))
  elseif status === :maxiter
    _styled(p, "  ⚠ ", :yellow; bold = true)
    println(p.io, "maximum number of iterations ($k), objective ", @sprintf("%.6e", f))
  else
    _styled(p, "  ⚠ ", :yellow; bold = true)
    println(p.io, "stalled: no acceptable step after $k iterations, objective ", @sprintf("%.6e", f))
  end
end

function _print_failure(p::_Printer, msg, k)
  p.level >= 1 || return
  p.level >= 2 && (p.header = false; _rule(p))
  _styled(p, "  ✗ ", :red; bold = true)
  println(p.io, "failed after $k accepted iterations: ", msg)
end

# solves: (forward, adjoint, filter) solved in this call, the setup and the final synchronisation included
function _print_summary(p::_Printer, solves, iterations)
  p.level >= 1 || return
  t = time() - p.t0
  println(p.io, @sprintf("  solves: forward %d, adjoint %d, filter %d · time %.2f s (%.3f s per iteration)",
                         solves..., t, t / max(iterations, 1)))
end
