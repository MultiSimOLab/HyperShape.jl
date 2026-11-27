# Runs the validation cases one after another, each in its own Julia process (the threads of this one), and joins
# their fixed report tables into results/report.csv and results/report.md (with the figures in the same folder).
#
#   julia --project=validation -t 4 validation/run_all.jl [case ...]      # all: about TOTAL_MIN min (Mac M5 Max, 4 threads)
#
# The cases are the scripts of this folder (without .jl); with no argument, all of CASES. The output folder is
# validation/results, or ENV["HYPERSHAPE_RESULTS"]. Each case writes its output (stdout and stderr, in order) to
# results/<case>.log. A case fails when its process exits with an error: an exception, or a check of the case that
# does not hold (the scripts exit with code 1). The others still run; run_all exits with code 1 if any failed.
isdefined(@__MODULE__, :write_report) || include(joinpath(@__DIR__, "common", "report.jl"))

const CASES = ["mbb_top88", "mbb_simpl", "cantilever3d", "bridge_takezawa", "mesh_double_well", "otomori_levelset88"]

"""
    run_all(cases = CASES; dir = results_dir(), scripts = @__DIR__, project = @__DIR__, known = CASES, io = stdout)

Runs `scripts/<case>.jl` for each case (`--project = project`, the threads of this process) with its output in
`dir/<case>.log`, and joins the tables `dir/<case>.csv` and `.md` of the cases that succeeded into `dir/report.csv` and
`dir/report.md`. Returns the cases that failed.
"""
function run_all(cases = CASES; dir = results_dir(), scripts = @__DIR__, project = @__DIR__, known = CASES, io = stdout)
  unknown = setdiff(cases, known)
  isempty(unknown) || throw(ArgumentError("run_all: unknown case(s) $(join(unknown, ", ")); the cases are $(join(known, ", "))"))
  mkpath(dir)
  julia = Base.julia_cmd()
  failed = String[]
  for case in cases
    script = joinpath(scripts, case * ".jl")
    log = joinpath(dir, case * ".log")
    println(io, "== ", case, " (log: ", log, ")")
    flush(io)
    cmd = addenv(`$julia --project=$project -t $(Threads.nthreads()) $script`, "HYPERSHAPE_RESULTS" => dir)
    # one stream for stdout and stderr: both in order in the same file
    t = @elapsed ok = open(f -> success(pipeline(cmd; stdout = f, stderr = f)), log, "w")
    @printf(io, "   %s in %.0f s\n", ok ? "done" : "FAILED", t)
    ok || push!(failed, case)
  end
  # join the tables of the cases that succeeded
  done = [c for c in cases if !(c in failed) && isfile(joinpath(dir, c * ".csv"))]
  open(joinpath(dir, "report.csv"), "w") do f
    for (k, c) in enumerate(done)
      lines = readlines(joinpath(dir, c * ".csv"))
      foreach(l -> println(f, l), k == 1 ? lines : lines[2:end])
    end
  end
  open(joinpath(dir, "report.md"), "w") do f
    println(f, "# Validation report\n\n", machine(), " · commit ", commit(), "\n")
    for c in done
      println(f, read(joinpath(dir, c * ".md"), String))
    end
    isempty(failed) || println(f, "Failed: ", join(failed, ", "), " (see the logs)")
  end
  println(io, "report: ", joinpath(dir, "report.md"), isempty(failed) ? "" : "; FAILED: " * join(failed, ", "))
  failed
end

if abspath(PROGRAM_FILE) == @__FILE__
  isempty(run_all(isempty(ARGS) ? CASES : ARGS)) || exit(1)
end
