# The fixed report table of the validation cases and benchmarks: one row per run, written to CSV and Markdown
# (validation/results/<name>.csv and .md; run_all.jl joins them). Columns (see the Benchmarks page of the docs):
#
#   case, variant          what was run (problem and mesh; design, method and options)
#   J                      the objective of the method on its final design (`result.objective`, unscaled)
#   J_bin                  the objective of the design read as 0/1 per cell (χ ≥ ½), the same mesh and loads,
#                          E = Emin + (1 − Emin) χ_bin (measures.jl, `binary_evaluator`): compares methods
#   ref, basis, vs_ref     the reference value, the column it is compared with (J or J_bin) and the deviation
#   V, V_bin               the volume fraction that the constraint sees, the fraction of solid cells
#   grey                   mean of 4χ(1 − χ) over the cell means
#   perimeter, holes       of the 0/1 design: interior solid–void faces (domain units), interior holes
#   iterations, status, reason
#   forward, adjoint, filter   solves of the state, the adjoint and the filter (sums over the history)
#   step                   solves of the step of reaction–diffusion: not in the public history yet (TODO: PR
#                          step-solves); "–"
#   time, t_per_state      seconds of `solve!` (no compilation) and that time divided by the state solves
using Printf

const REPORT_COLUMNS = (:case, :variant, :J, :J_bin, :ref, :basis, :vs_ref, :V, :V_bin, :grey, :perimeter, :holes,
                        :iterations, :status, :reason, :forward, :adjoint, :filter, :step, :time, :t_per_state)

"directory of the results: ENV[\"HYPERSHAPE_RESULTS\"] or validation/results"
results_dir() = get(ENV, "HYPERSHAPE_RESULTS", joinpath(dirname(@__DIR__), "results"))

machine() = string(Sys.cpu_info()[1].model, ", ", Threads.nthreads(), " threads, Julia ", VERSION)
function commit()
  try
    readchomp(`git -C $(dirname(@__DIR__)) rev-parse --short HEAD`)
  catch
    "unknown"
  end
end

_vs(x, ref) = (x === missing || ref === missing) ? missing : x / ref - 1

"""
    report_row(case, variant, c; ref = missing, basis = :J, J_bin = missing, grey_cells = c.cells, kw...)

The row of a run of HyperShape: `c` is what the problems of problems.jl return (`result`, `time`, `cells`, `n`,
`hcell`, `vf`, `jbin`). `J_bin` is computed with `c.jbin` unless given; `grey_cells` are the cell means used for the
grey (the physical field), when `c.cells` is a 0/1 design. Keywords override any column.
"""
function report_row(case, variant, c; ref = missing, basis = :J, J_bin = missing, grey_cells = c.cells, kw...)
  r = c.result
  h = r.history
  k = h.n
  Jb = J_bin === missing ? c.jbin(c.cells) : J_bin
  J = r.objective
  fw = sum(view(h.forward, 1:k))
  row = (; case, variant, J, J_bin = Jb, ref, basis, vs_ref = _vs(basis === :J ? J : Jb, ref),
         V = c.vf * (1 + h.constraints.volume[end]), V_bin = binary_volume(c.cells), grey = grey(grey_cells),
         perimeter = perimeter(c.cells, c.n, c.hcell), holes = interior_holes(reshape(c.cells, c.n)),
         iterations = r.iterations, status = r.status, reason = r.reason, forward = fw,
         adjoint = sum(view(h.adjoint, 1:k)), filter = sum(view(h.filter, 1:k)), step = missing, time = c.time,
         t_per_state = c.time / fw)
  merge(row, kw)
end

"""
    reference_row(case, variant; J, cells = nothing, n, hcell, jbin = nothing, J_bin = missing, iterations = missing, kw...)

The row of a reference (a port, a code run in Octave, a published value): the measures of its design when its cell
values are given, "–" for what it does not have.
"""
function reference_row(case, variant; J, cells = nothing, n = nothing, hcell = nothing, jbin = nothing,
                       J_bin = missing, V = missing, iterations = missing, kw...)
  m = missing
  have = cells !== nothing
  Jb = J_bin !== missing ? J_bin : (have && jbin !== nothing ? jbin(cells) : m)
  row = (; case, variant, J, J_bin = Jb, ref = m, basis = :J, vs_ref = m, V,
         V_bin = have ? binary_volume(cells) : m, grey = have ? grey(cells) : m,
         perimeter = have ? perimeter(cells, n, hcell) : m, holes = have ? interior_holes(reshape(cells, n)) : m,
         iterations, status = m, reason = m, forward = m, adjoint = m, filter = m, step = m, time = m, t_per_state = m)
  merge(row, kw)
end

_csv(x::Missing) = ""
_csv(x::Nothing) = ""
_csv(x::AbstractFloat) = @sprintf("%.10g", x)
_csv(x) = (s = string(x); occursin(r"[,\"]", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s)

_md(x, f) = x === missing || x === nothing ? "–" : f(x)
_g(x) = @sprintf("%.6g", x)
function _md_row(r)
  vs = r.vs_ref === missing ? "–" : @sprintf("%+.2f %% (%s)", 100 * r.vs_ref, r.basis === :J ? "J" : "J_bin")
  its = r.iterations === missing ? "–" :
        r.status === missing ? string(r.iterations) :
        string(r.iterations, " (", r.status, r.reason === nothing || r.reason === missing ? "" : ", $(r.reason)", ")")
  sol = r.forward === missing ? "–" : string(r.forward, "/", r.adjoint, "/", r.filter, "/", _md(r.step, string))
  string("| ", r.case, " | ", r.variant, " | ", _md(r.J, _g), " | ", _md(r.J_bin, _g), " | ", _md(r.ref, _g), " | ", vs,
         " | ", _md(r.V, x -> @sprintf("%.4f", x)), " | ", _md(r.V_bin, x -> @sprintf("%.4f", x)), " | ",
         _md(r.grey, x -> @sprintf("%.3f", x)), " | ", _md(r.perimeter, x -> @sprintf("%.4g", x)), " | ",
         _md(r.holes, string), " | ", its, " | ", sol, " | ", _md(r.time, x -> @sprintf("%.1f", x)), " | ",
         _md(r.t_per_state, x -> @sprintf("%.3g", x)), " |")
end

const MD_HEADER = "| case | variant | J | J_bin | ref | vs ref | V | V_bin | grey | perimeter | holes | iterations | " *
                  "solves state/adjoint/filter/step | time (s) | time / state solve (s) |\n" *
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"

"the Markdown table of the rows"
markdown_table(rows) = join([MD_HEADER; [_md_row(r) for r in rows]], "\n")

"""
    write_report(name, rows; title = name, dir = results_dir(), io = stdout)

Writes `dir/name.csv` (one row per run, the columns of `REPORT_COLUMNS` plus machine, commit and date) and
`dir/name.md` (title, machine line and the Markdown table), and prints the table to `io`.
"""
function write_report(name, rows; title = name, dir = results_dir(), io = stdout)
  mkpath(dir)
  m, c, d = machine(), commit(), string(Base.Libc.strftime("%Y-%m-%d %H:%M", time()))
  open(joinpath(dir, name * ".csv"), "w") do f
    println(f, join((string.(REPORT_COLUMNS)..., "machine", "commit", "date"), ","))
    for r in rows
      println(f, join((_csv.(getproperty.(Ref(r), REPORT_COLUMNS))..., _csv(m), _csv(c), _csv(d)), ","))
    end
  end
  md = string("### ", title, "\n\n", m, " · commit ", c, " · ", d, "\n\n", markdown_table(rows), "\n")
  write(joinpath(dir, name * ".md"), md)
  println(io, "\nFixed report table (", joinpath(dir, name * ".md"), "):\n\n", markdown_table(rows))
  nothing
end

# one line of the short table that the scripts print (kept as it was before the fixed table)
row(name, r, ref = NaN) = @sprintf("| %s | %.6g | %s | %d | %s | %.1f s |", name, r.result.objective,
                                   isnan(ref) ? "—" : @sprintf("%+.2f %%", 100 * (r.result.objective / ref - 1)),
                                   r.result.iterations, r.result.status, r.time)
