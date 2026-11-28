# The solves of the step of ReactionDiffusion in the history (history.step_solves): per iteration with and without the
# unit response φ̌, the double well (implicit and explicit), the screen output, the checkpoint (format 4, and a real
# checkpoint of format 3 written by an earlier version), a rebuild and allocations. Uses the helpers of
# reaction_diffusion.jl and double_well.jl (rd_cantilever, rd_factories, rd_start, rd, dw_cantilever, dw).
using Test
using TOML
using Gridap
using HyperShape

struct _OneMonitor end
HyperShape.monitor_value(::_OneMonitor) = 1.0

@testset "ReactionDiffusion: solves of the step in the history" begin

  @testset "1–2. without the well: one solve per iteration; φ̌ of init only in the summary" begin
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
  end

  @testset "4. double well, explicit scheme: no solve of the step, the column is shown" begin
    # (the implicit scheme, n solves per iteration and 1 after the settling: test 13 of double_well.jl)
    io = IOBuffer()
    r = solve!(dw_cantilever(; nx = 20, ny = 10).problem, dw(; scheme = :explicit, substeps = 20); maxiter = 3,
               stop = nothing, verbose = 2, io)
    out = String(take!(io))
    @test r.history.step_solves == zeros(Int, 3) && r.state.nsolves == 0
    @test occursin("f/a/F/s", out) && occursin(r"\s\d+/\d+/\d+/0\s", out) && occursin(", step 0 · time", out)
  end

  @testset "12. a rebuild: the history as with FixedMesh, the summary adds φ̌ of each init" begin
    io1, io2 = IOBuffer(), IOBuffer()
    f1 = solve!(rd_factories(), rd(); maxiter = 15, stop = nothing, verbose = 1, io = io1)
    f2 = solve!(rd_factories(), rd(); maxiter = 15, stop = nothing, verbose = 1, io = io2,
                discretization = RDM.RebuildCheck(every = 6))
    @test f1.history.step_solves == f2.history.step_solves == ones(Int, 15)
    @test occursin(", step 16 · time", String(take!(io1)))
    @test occursin(", step 18 · time", String(take!(io2)))      # the inits at 0, 6 and 12 each solve φ̌
  end

  @testset "13. checkpoint of format 4: the column step_solves, read back" begin
    dir = mktempdir()
    r = solve!(rd_cantilever(; nx = 20, ny = 10).problem, rd(); maxiter = 4, stop = nothing, verbose = 0,
               checkpoint = Checkpoint(filepath = dir, every = 2))
    @test TOML.parsefile(joinpath(dir, "checkpoint.toml"))["hypershape_checkpoint"] == 4
    head = split(first(readlines(joinpath(dir, "history_4.csv"))), ",")
    @test head[11] == "step_solves" && head[12] == "constraint:volume"
    st = load_checkpoint(dir)
    @test st.format == 4 && st.history.step_solves == r.history.step_solves == ones(Int, 4)
    @test r.problem.state[].format == 4
    # the name of the column is taken: a monitor or a continued parameter called step_solves is refused
    @test_throws ArgumentError solve!(rd_cantilever(; nx = 20, ny = 10).problem, rd(); maxiter = 2, verbose = 0,
                                      monitors = (step_solves = _OneMonitor(),))
    @test_throws ArgumentError Continuation(step_solves = Parameter(1.0) => [1.0, 2.0]; every = 2)
  end

  @testset "14. a real checkpoint of format 3 (HyperShape main 043b388): read, resumed bit for bit" begin
    src = joinpath(@__DIR__, "..", "data", "checkpoint_v3")
    dir = mktempdir()
    for f in readdir(src)
      cp(joinpath(src, f), joinpath(dir, f))
    end
    @test TOML.parsefile(joinpath(dir, "checkpoint.toml"))["hypershape_checkpoint"] == 3
    @test !occursin("step_solves", first(readlines(joinpath(dir, "history_6.csv"))))
    st = load_checkpoint(dir)
    @test st.format == 3 && st.iteration == 6 && st.history.step_solves == zeros(Int, 6)
    # the uninterrupted run with this version, and the run resumed from the old checkpoint
    r0 = solve!(rd_cantilever(; nx = 20, ny = 10).problem, rd(); maxiter = 12, stop = nothing, verbose = 0)
    @test r0.history.objective[1:6] == st.history.objective && r0.history.constraints.volume[1:6] == st.history.constraints.volume
    io = IOBuffer()
    new = mktempdir()
    r1 = solve!(rd_cantilever(; nx = 20, ny = 10).problem, rd(); maxiter = 12, stop = nothing, verbose = 1, io,
                resume = dir, checkpoint = Checkpoint(filepath = new, every = 12))
    out = String(take!(io))
    @test get_values(r1.z) == get_values(r0.z) && r1.history.objective == r0.history.objective
    @test r1.history.constraints.volume == r0.history.constraints.volume
    @test r1.history.step_solves == [zeros(Int, 6); r0.history.step_solves[7:12]]
    @test occursin("checkpoint of format version 3: the step solves of its iterations were not recorded", out)
    @test TOML.parsefile(joinpath(new, "checkpoint.toml"))["hypershape_checkpoint"] == 4
    @test load_checkpoint(new).history.step_solves == r1.history.step_solves
    # versions 1 and 2 and no version are still refused
    for (from, to) in (("hypershape_checkpoint = 3", "hypershape_checkpoint = 2"), ("hypershape_checkpoint = 2", "format = 3"))
      tfile = joinpath(dir, "checkpoint.toml")
      write(tfile, replace(read(tfile, String), from => to))
      err = try load_checkpoint(dir); nothing catch e; e end
      @test err isa ArgumentError && occursin("reads versions 3 and 4", err.msg)
    end
  end

  @testset "16. the count of the step does not allocate" begin
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
