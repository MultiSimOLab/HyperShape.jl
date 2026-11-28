# The solves of the step of ReactionDiffusion in the history (history.step_solves): per iteration with and without the
# unit response φ̌, held values, a solve of the user, a jump of τ/Δt, the double well (explicit), the screen output,
# rejected steps and resumes, a rebuild, failures, the checkpoint (format 4, and a real checkpoint of format 3 written
# by an earlier version) and allocations. Uses the helpers of reaction_diffusion.jl and double_well.jl (rd_cantilever,
# rd_factories, rd_start, rd, dw_cantilever, dw).
using Test
using TOML
using Gridap
using HyperShape

struct _OneMonitor end
HyperShape.monitor_value(::_OneMonitor) = 1.0

# a copy of a checkpoint folder
function _copy_ckpt(src)
  dir = mktempdir()
  foreach(f -> cp(joinpath(src, f), joinpath(dir, f)), readdir(src))
  dir
end
# the checkpoint folder rewritten in the other format: v4 → v3 drops the column step_solves, v3 → v4 adds it (zeros)
function _convert_ckpt!(dir, to)
  t = joinpath(dir, "checkpoint.toml")
  meta = TOML.parsefile(t)
  h = joinpath(dir, meta["history_file"])
  rows = [split(l, ",") for l in readlines(h)]
  if to == 3
    rows = [r[[1:10; 12:end]] for r in rows]
  else
    rows = [[r[1:10]; (i == 1 ? "step_solves" : "0"); r[11:end]] for (i, r) in enumerate(rows)]
  end
  write(h, join((join(r, ",") for r in rows), "\n") * "\n")
  write(t, replace(read(t, String), "hypershape_checkpoint = $(to == 3 ? 4 : 3)" => "hypershape_checkpoint = $to"))
  dir
end
const _V3_EVENT = "checkpoint of format version 3: the step solves of its iterations were not recorded"

@testset "ReactionDiffusion: solves of the step in the history" begin

  @testset "without the well: one solve per iteration; φ̌ of init only in the summary" begin
    # no held values (unit: φ̌ = Δt, nothing solved)
    r = solve!(rd_cantilever(; nx = 20, ny = 10, frozen = false).problem, rd(); maxiter = 5, stop = nothing, verbose = 0)
    @test r.state.unit && r.history.step_solves == ones(Int, 5) && sum(r.history.step_solves) == r.state.nsolves
    @test RDM.step_solves(rd().step, r.state) == r.state.nsolves
    # a frozen zone under the load: φ̌ solved at init (the summary counts it, the rows do not)
    io = IOBuffer()
    r = solve!(rd_cantilever(; nx = 20, ny = 10).problem, rd(); maxiter = 5, stop = nothing, verbose = 2, io)
    out = String(take!(io))
    @test !r.state.unit && r.history.step_solves == ones(Int, 5) && r.state.nsolves == sum(r.history.step_solves) + 1
    @test occursin("f/a/F/s", out) && occursin("1/0/0/1", out)
    @test occursin(", step 6 · time", out) && occursin("reaction–diffusion: 1 factorisation,", out)
    # values held by phi_bc, and a solve of the user (its calls are the solves of the summary)
    r = solve!(rd_cantilever(; nx = 20, ny = 10, frozen = false).problem, rd(; phi_bc = :zero_boundary); maxiter = 4,
               stop = nothing, verbose = 0)
    @test !r.state.unit && r.history.step_solves == ones(Int, 4) && r.state.nsolves == 5
    calls = Ref(0)
    us = (A, b, x, ctx, stale) -> (calls[] += 1; A \ b)
    io = IOBuffer()
    r = solve!(rd_cantilever(; nx = 20, ny = 10).problem, rd(; solve = us); maxiter = 4, stop = nothing, verbose = 1, io)
    @test r.history.step_solves == ones(Int, 4) && calls[] == r.state.nsolves == 5
    @test occursin(", step 5 · time", String(take!(io)))
  end

  @testset "a jump of τ and Δt by a continuation: φ̌ of the new factorisation in the row of the jump; resume at it" begin
    function contrun(; maxiter, kw...)
      pτ, pΔt = Parameter(2e-4), Parameter(0.1)
      solve!(rd_cantilever(; nx = 20, ny = 10).problem, ReactionDiffusion(; τ = pτ, Δt = pΔt);
             maxiter, stop = nothing, verbose = 0,
             continuation = Continuation(τ = pτ => [2e-4, 1e-3], Δt = pΔt => [0.1, 0.05]; every = 3), kw...)
    end
    r0 = contrun(; maxiter = 6)
    @test r0.history.step_solves == [1, 1, 1, 2, 1, 1]
    for k in (3, 4)                  # the checkpoint at the end of the first stage and at the jump
      d = mktempdir()
      contrun(; maxiter = k, checkpoint = Checkpoint(filepath = d, every = k))
      r1 = contrun(; maxiter = 6, resume = d)
      @test r1.history.step_solves == r0.history.step_solves && get_values(r1.z) == get_values(r0.z)
    end
  end

  @testset "double well, explicit scheme: no solve of the step, the column is shown" begin
    # (the implicit scheme, n solves per iteration and 1 after the settling: test 13 of double_well.jl)
    io = IOBuffer()
    r = solve!(dw_cantilever(; nx = 20, ny = 10).problem, dw(; scheme = :explicit, substeps = 20); maxiter = 3,
               stop = nothing, verbose = 2, io)
    out = String(take!(io))
    @test r.history.step_solves == zeros(Int, 3) && r.state.nsolves == 0
    @test occursin("f/a/F/s", out) && occursin(r"\s\d+/\d+/\d+/0\s", out) && occursin(", step 0 · time", out)
    # the column is wide enough for 20 substeps: the rows end aligned with the header
    io = IOBuffer()
    solve!(dw_cantilever(; nx = 20, ny = 10).problem, dw(; substeps = 20); maxiter = 2, stop = nothing, verbose = 2, io)
    lines = split(String(take!(io)), '\n')
    endcol(l, p) = textwidth(l[1:last(findfirst(p, l))])
    head = first(filter(l -> occursin("f/a/F/s", l), lines))
    rows = filter(l -> occursin(r"\d+/\d+/\d+/20\s", l), lines)
    @test length(rows) == 2 && all(l -> endcol(l, r"\d+/\d+/\d+/20") == endcol(head, "f/a/F/s"), rows)
  end

  @testset "a rejected step, and a resume right after it: the φ̌ of the return to Δt goes to init" begin
    flag = ctx -> !(ctx.iteration == 3 && ctx.attempt == 1)
    r0 = solve!(rd_cantilever(; nx = 20, ny = 10, solve_flag = flag).problem, rd(); maxiter = 6, stop = nothing, verbose = 0)
    @test r0.history.step_solves == [1, 1, 3, 2, 1, 1]
    d = mktempdir()
    solve!(rd_cantilever(; nx = 20, ny = 10, solve_flag = flag).problem, rd(); maxiter = 3, stop = nothing, verbose = 0,
           checkpoint = Checkpoint(filepath = d, every = 3))
    io = IOBuffer()
    r1 = solve!(rd_cantilever(; nx = 20, ny = 10, solve_flag = flag).problem, rd(); maxiter = 6, stop = nothing,
                verbose = 1, io, resume = d)
    @test get_values(r1.z) == get_values(r0.z) && r1.history.objective == r0.history.objective
    @test r1.history.step_solves == [1, 1, 3, 1, 1, 1]           # its init factorises at Δt: one solve less in row 4
    @test occursin(", step 4 · time", String(take!(io)))          # φ̌ of init and the steps of 4–6
  end

  @testset "a rebuild: the history as with FixedMesh, the summary adds φ̌ of each init" begin
    io1, io2 = IOBuffer(), IOBuffer()
    f1 = solve!(rd_factories(), rd(); maxiter = 15, stop = nothing, verbose = 1, io = io1)
    f2 = solve!(rd_factories(), rd(); maxiter = 15, stop = nothing, verbose = 1, io = io2,
                discretization = RDM.RebuildCheck(every = 6))
    @test f1.history.step_solves == f2.history.step_solves == ones(Int, 15)
    @test occursin(", step 16 · time", String(take!(io1)))
    @test occursin(", step 18 · time", String(take!(io2)))      # the inits at 0, 6 and 12 each solve φ̌
  end

  @testset "a failure: the summary has the step solves" begin
    io = IOBuffer()
    @test_throws OptimizationFailure solve!(rd_cantilever(; nx = 20, ny = 10, solve_flag = ctx -> ctx.iteration < 3).problem,
                                            rd(); maxiter = 5, stop = nothing, verbose = 1, io)
    @test occursin(r"solves: forward \d+, adjoint 0, filter 0, step \d+ · time", String(take!(io)))
  end

  @testset "checkpoint of format 4: the column step_solves, read back; the name is taken" begin
    dir = mktempdir()
    r = solve!(rd_cantilever(; nx = 20, ny = 10).problem, rd(); maxiter = 4, stop = nothing, verbose = 0,
               checkpoint = Checkpoint(filepath = dir, every = 2))
    @test TOML.parsefile(joinpath(dir, "checkpoint.toml"))["hypershape_checkpoint"] == 4
    head = split(first(readlines(joinpath(dir, "history_4.csv"))), ",")
    @test head[11] == "step_solves" && head[12] == "constraint:volume"
    st = load_checkpoint(dir)
    @test st.history.step_solves == r.history.step_solves == ones(Int, 4)
    @test !RDM._unrecorded(st) && !RDM._unrecorded(r.problem.state[])
    # OptState keeps the fields of before (nothing about the format of the checkpoint)
    @test fieldcount(OptState) == 18 && !hasfield(OptState, :format)
    # a checkpoint.toml of version 3 beside a history with the column step_solves: an error
    t = joinpath(dir, "checkpoint.toml")
    write(t, replace(read(t, String), "hypershape_checkpoint = 4" => "hypershape_checkpoint = 3"))
    err = try load_checkpoint(dir); nothing catch e; e end
    @test err isa ArgumentError && occursin("format version 3 and its history has a column step_solves", err.msg)
    # a monitor or a continued parameter called step_solves is refused
    @test_throws ArgumentError solve!(rd_cantilever(; nx = 20, ny = 10).problem, rd(); maxiter = 2, verbose = 0,
                                      monitors = (step_solves = _OneMonitor(),))
    @test_throws ArgumentError Continuation(step_solves = Parameter(1.0) => [1.0, 2.0]; every = 2)
  end

  @testset "a real checkpoint of format 3 (HyperShape main 043b388): read and resumed" begin
    src = joinpath(@__DIR__, "..", "data", "checkpoint_v3")
    dir = _copy_ckpt(src)
    @test TOML.parsefile(joinpath(dir, "checkpoint.toml"))["hypershape_checkpoint"] == 3
    @test !occursin("step_solves", first(readlines(joinpath(dir, "history_6.csv"))))
    st = load_checkpoint(dir)
    @test RDM._unrecorded(st) && st.iteration == 6 && st.history.step_solves == zeros(Int, 6)
    # the resume from the old folder and from the same folder written as version 4: the same bits on any machine
    io = IOBuffer()
    new = mktempdir()
    r1 = solve!(rd_cantilever(; nx = 20, ny = 10).problem, rd(); maxiter = 12, stop = nothing, verbose = 1, io,
                resume = dir, checkpoint = Checkpoint(filepath = new, every = 12))
    out = String(take!(io))
    io4 = IOBuffer()
    r4 = solve!(rd_cantilever(; nx = 20, ny = 10).problem, rd(); maxiter = 12, stop = nothing, verbose = 1, io = io4,
                resume = _convert_ckpt!(_copy_ckpt(src), 4))
    @test get_values(r1.z) == get_values(r4.z) && r1.history.objective == r4.history.objective
    @test r1.history.constraints == r4.history.constraints && r1.history.step_solves == r4.history.step_solves
    @test occursin(_V3_EVENT, out) && !occursin(_V3_EVENT, String(take!(io4)))
    # against the uninterrupted run with this version (the stored files were written on another machine: a tolerance)
    r0 = solve!(rd_cantilever(; nx = 20, ny = 10).problem, rd(); maxiter = 12, stop = nothing, verbose = 0)
    @test r0.history.objective[1:6] ≈ st.history.objective rtol = 1e-10
    @test isapprox(r1.history.objective, r0.history.objective; rtol = 1e-10) && isapprox(get_values(r1.z), get_values(r0.z); rtol = 1e-8)
    @test r1.history.step_solves == [zeros(Int, 6); r0.history.step_solves[7:12]]
    @test TOML.parsefile(joinpath(new, "checkpoint.toml"))["hypershape_checkpoint"] == 4
    @test load_checkpoint(new).history.step_solves == r1.history.step_solves
    # a second resume, from the state of that run or from the checkpoint it wrote: the zeros stay, no notice
    for res in (r1.problem.state[], new)
      io = IOBuffer()
      r2 = solve!(rd_cantilever(; nx = 20, ny = 10).problem, rd(); maxiter = 14, stop = nothing, verbose = 1, io, resume = res)
      @test r2.history.step_solves[1:6] == zeros(Int, 6) && !occursin(_V3_EVENT, String(take!(io)))
    end
    # versions 1 and 2 and no version are still refused
    for (from, to) in (("hypershape_checkpoint = 3", "hypershape_checkpoint = 2"), ("hypershape_checkpoint = 2", "format = 3"))
      tfile = joinpath(dir, "checkpoint.toml")
      write(tfile, replace(read(tfile, String), from => to))
      err = try load_checkpoint(dir); nothing catch e; e end
      @test err isa ArgumentError && occursin("reads versions 3 and 4", err.msg)
    end
  end

  @testset "format 3 with continued parameters and monitors; a density method has no notice" begin
    # a checkpoint of this version rewritten as version 3 (without the column): the columns after it are read
    pτ = Parameter(2e-4)
    d = mktempdir()
    mk() = solve!(rd_cantilever(; nx = 20, ny = 10).problem, ReactionDiffusion(; τ = pτ);
                  maxiter = 4, stop = nothing, verbose = 0, monitors = (one = _OneMonitor(),),
                  continuation = Continuation(τ = pτ => [2e-4, 1e-3]; every = 2), checkpoint = Checkpoint(filepath = d, every = 4))
    r = mk()
    st4 = load_checkpoint(d)
    st3 = load_checkpoint(_convert_ckpt!(_copy_ckpt(d), 3))
    @test RDM._unrecorded(st3) && st3.history.step_solves == zeros(Int, 4)
    @test st3.history.continuation == st4.history.continuation == r.history.continuation
    @test st3.history.monitors == st4.history.monitors && st3.history.constraints == st4.history.constraints
    # density: no notice (its 0 is exact), the same design as from version 4
    c = rd_cantilever(; nx = 20, ny = 10, design = :density, frozen = false)
    dd = mktempdir()
    solve!(c.problem, DensityMethod(OC(); metric = Euclidean()); maxiter = 3, stop = nothing, verbose = 0,
           checkpoint = Checkpoint(filepath = dd, every = 3))
    rerun(res) = (io = IOBuffer();
                r = solve!(rd_cantilever(; nx = 20, ny = 10, design = :density, frozen = false).problem,
                           DensityMethod(OC(); metric = Euclidean()); maxiter = 5, stop = nothing, verbose = 1, io, resume = res);
                (r, String(take!(io))))
    ra, oa = rerun(_convert_ckpt!(_copy_ckpt(dd), 3))
    rb, ob = rerun(dd)
    @test get_values(ra.z) == get_values(rb.z) && ra.history.step_solves == zeros(Int, 5)
    @test !occursin(_V3_EVENT, oa) && occursin(r"solves: forward \d+, adjoint \d+, filter \d+ · time", oa)
  end

  @testset "the count of the step does not allocate" begin
    c = rd_cantilever(; nx = 20, ny = 10)
    m = rd()
    fo, z, s = rd_start(c, m)
    RDM._step_count(m.step, s)
    @test (@allocated RDM._step_count(m.step, s)) == 0
    h = RDM.OptHistory(4)
    RDM._record!(h, (1, 1.0, NaN, NaN, 0.1, 0.2, 1, 1, 0, 0), (), (), (), (), (), (), 3)
    @test (@allocated RDM._record!(h, (2, 1.0, NaN, NaN, 0.1, 0.2, 1, 1, 0, 0), (), (), (), (), (), (), 5)) == 0
    @test h.step_solves[1:2] == [3, 5]
  end
end
