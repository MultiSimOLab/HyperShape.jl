# Screen output of the optimisation loop, in the style of HyperFEM's Newton log: a banner, a row per iteration,
# events (continuation jumps, rejected steps, stalls) as indented coloured lines, a closing ✓/⚠/✗ line and a summary.
# Levels: 0 silent; 1 banner, events, closing line and summary; 2 (default) also a row per iteration.
# Colours only on a terminal (`printstyled` writes plain text to files and buffers).

mutable struct _Printer{IO_}
  io::IO_
  level::Int
  t0::Float64
  tlast::Float64
  rows::Int
end

_verbose_level(v::Bool) = v ? 2 : 0
_verbose_level(v::Integer) = (0 <= v <= 2 || throw(ArgumentError("solve!: verbose must be 0, 1, 2 or a Bool")); Int(v))
_verbose_level(v) = throw(ArgumentError("solve!: verbose must be 0, 1, 2 or a Bool"))

_Printer(io, verbose) = _Printer(io, _verbose_level(verbose), time(), time(), 0)

_styled(p::_Printer, s, color = :normal; bold = false) = printstyled(p.io, s; color, bold)

_e(x) = isfinite(x) ? @sprintf("%9.2e", x) : "        —"

function _rule(p::_Printer, width = 100)
  _styled(p, "  " * "─"^width * "\n", :light_black)
end

function _describe_stop(stop)
  stop === nothing && return "maxiter only"
  parts = String[]
  stop.change !== nothing && push!(parts, "change < $(stop.change)")
  stop.kkt !== nothing && push!(parts, "kkt < $(stop.kkt)")
  stop.objective !== nothing && push!(parts, "Δobjective < $(stop.objective) over $(stop.window)")
  isempty(parts) ? "maxiter only" : join(parts, ", ")
end

_metric_name(m::Euclidean) = "Euclidean"
_metric_name(m::LumpedMass) = "lumped mass"
_metric_name(m::NamedTuple) = join(("$k: $(_metric_name(v))" for (k, v) in pairs(m)), ", ")

function _print_banner(p::_Printer, problem, method, maxiter, stop, schedule, resume)
  p.level >= 1 || return
  opt = method.optimizer
  z = problem.z
  title = " HyperShape · SIMP($(nameof(typeof(opt)))) "
  _styled(p, "━━" * title * "━"^max(3, 98 - length(title)) * "\n", :cyan; bold = true)
  blocks = join(("$n ($(length(r)))" for (n, r) in zip(z.names, z.ranges)), ", ")
  nc = length(problem.linear_constraints)
  println(p.io, "  design variables $(length(z)): $blocks · constraint: ", nc == 0 ? "none" : "linear", " · metric: ",
          _metric_name(method.metric))
  if schedule !== nothing
    for (n, v) in zip(schedule.names, schedule.values)
      println(p.io, "  continuation $n: ", join((_g(x) for x in v), " → "), " (every $(schedule.every) iterations)")
    end
  end
  println(p.io, "  stop: ", _describe_stop(stop), " · maxiter $maxiter",
          resume === nothing ? "" : " · resumed at iteration $(resume.iteration)")
  _print_columns(p, schedule)
end

_g(x) = x == round(x) && abs(x) < 1e6 ? string(Int(x)) : @sprintf("%.3g", x)

function _print_columns(p::_Printer, schedule)
  p.level >= 2 || return
  cont = schedule === nothing ? "" : join((@sprintf("%6s", n) for n in schedule.names))
  _rule(p)
  _styled(p, @sprintf("  %5s  %13s  %9s  %10s  %9s  %9s  %9s  %9s%s  %9s  %7s\n", "it", "objective", "Δobj",
                       "constraint", "kkt", "change", "step", "μ", cont, "f/a/F", "t[s]"); bold = true)
  _rule(p)
end

function _print_row(p::_Printer, hist::OptHistory)
  p.level >= 2 || return
  k = hist.n
  t = time()
  dt = t - p.tlast
  p.tlast = t
  f = hist.objective[k]
  df = k > 1 ? (f - hist.objective[k-1]) / max(abs(hist.objective[k-1]), floatmin()) : NaN
  g = hist.constraint[k]
  cont = join((@sprintf("%6s", _g(hist.continuation[j, k])) for j in axes(hist.continuation, 1)))
  solves = "$(hist.forward[k])/$(hist.adjoint[k])/$(hist.filter[k])"
  @printf(p.io, "  %5d  %13.6e  %9s  ", hist.iteration[k], f, isfinite(df) ? @sprintf("%+9.1e", df) : "        —")
  if isfinite(g)
    _styled(p, @sprintf("%10.2e", g), g > 1e-6 ? :red : :normal)
  else
    print(p.io, "         —")
  end
  @printf(p.io, "  %9s  %9s  %9s  %9s%s  %9s  %7.3f", _e(hist.kkt[k]), _e(hist.change[k]), _e(hist.step[k]),
          _e(hist.multiplier[k]), cont, solves, dt)
  hist.attempts[k] > 1 && _styled(p, "  ($(hist.attempts[k]) attempts)", :yellow)
  println(p.io)
  p.rows += 1
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
  p.level >= 2 && _rule(p)
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
  p.level >= 2 && _rule(p)
  _styled(p, "  ✗ ", :red; bold = true)
  println(p.io, "failed after $k accepted iterations: ", msg)
end

function _print_summary(p::_Printer, hist::OptHistory, k0)
  p.level >= 1 || return
  r = (k0+1):hist.n
  t = time() - p.t0
  println(p.io, @sprintf("  solves: forward %d, adjoint %d, filter %d · time %.2f s (%.3f s per iteration)",
                         sum(view(hist.forward, r); init = 0), sum(view(hist.adjoint, r); init = 0),
                         sum(view(hist.filter, r); init = 0), t, t / max(length(r), 1)))
end
