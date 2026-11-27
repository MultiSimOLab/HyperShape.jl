# Runs the validation cases one after another, each in its own Julia process (the threads of this one), and joins
# their fixed report tables into results/report.csv and results/report.md (with the figures in the same folder).
#
#   julia --project=validation -t 4 validation/run_all.jl [case ...]      # all: about 13 min (Mac M5 Max, 4 threads)
#
# The cases are the scripts of this folder (without .jl); with no argument, all of CASES. The output folder is
# validation/results, or ENV["HYPERSHAPE_RESULTS"]. A case that fails is reported and the others still run.
include(joinpath(@__DIR__, "common", "report.jl"))

const CASES = ["mbb_top88", "mbb_simpl", "cantilever3d", "bridge_takezawa", "mesh_double_well", "otomori_levelset88"]

function run_all(cases = CASES; dir = results_dir())
  mkpath(dir)
  julia = Base.julia_cmd()
  failed = String[]
  for case in cases
    case in CASES || throw(ArgumentError("run_all: unknown case $case; the cases are $(join(CASES, ", "))"))
    script = joinpath(@__DIR__, case * ".jl")
    log = joinpath(dir, case * ".log")
    println("== ", case, " (log: ", log, ")")
    flush(stdout)
    cmd = `$julia --project=$(@__DIR__) -t $(Threads.nthreads()) $script`
    t = @elapsed ok = success(pipeline(addenv(cmd, "HYPERSHAPE_RESULTS" => dir); stdout = log, stderr = log))
    @printf("   %s in %.0f s\n", ok ? "done" : "FAILED", t)
    ok || push!(failed, case)
  end
  # join the tables of the cases that ran
  csvs = [joinpath(dir, c * ".csv") for c in cases if isfile(joinpath(dir, c * ".csv")) && !(c in failed)]
  open(joinpath(dir, "report.csv"), "w") do io
    for (k, f) in enumerate(csvs)
      lines = readlines(f)
      foreach(l -> println(io, l), k == 1 ? lines : lines[2:end])
    end
  end
  open(joinpath(dir, "report.md"), "w") do io
    println(io, "# Validation report\n\n", machine(), " · commit ", commit(), "\n")
    for c in cases
      f = joinpath(dir, c * ".md")
      (isfile(f) && !(c in failed)) && println(io, read(f, String))
    end
    isempty(failed) || println(io, "Failed: ", join(failed, ", "), " (see the logs)")
  end
  println("report: ", joinpath(dir, "report.md"), isempty(failed) ? "" : "; FAILED: " * join(failed, ", "))
  failed
end

if abspath(PROGRAM_FILE) == @__FILE__
  isempty(run_all(isempty(ARGS) ? CASES : ARGS)) || exit(1)
end
