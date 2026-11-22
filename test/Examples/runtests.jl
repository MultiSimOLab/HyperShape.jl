# The scripts in examples/ run here at a small size, so the documented scripts keep working.
include(joinpath(@__DIR__, "..", "..", "examples", "mbb.jl"))
include(joinpath(@__DIR__, "..", "..", "examples", "mbb_factories.jl"))

@testset "Examples" begin
  @testset "MBB beam (examples/mbb.jl)" begin
    out = mktempdir()
    r = run_mbb(; nx = 30, ny = 10, maxiter = 12, out, verbose = 0)
    @test r.iterations == 12 && r.status === :maxiter
    @test r.history.objective[end] < r.history.objective[1]
    @test abs(r.history.constraints.volume[end]) < 1e-3
    @test isfile(joinpath(out, "checkpoint", "checkpoint.toml")) && isfile(joinpath(out, "vtk", "optimization.pvd"))
  end
  @testset "MBB beam with factories (examples/mbb_factories.jl): the run of mbb.jl, also with rebuilds" begin
    r0 = run_mbb(; nx = 30, ny = 10, maxiter = 12, out = mktempdir(), verbose = 0)
    for d in (FixedMesh(), HyperShape.Methods.RebuildCheck(every = 5))
      out = mktempdir()
      r = run_mbb_factories(; nx = 30, ny = 10, maxiter = 12, out, verbose = 0, discretization = d)
      @test r.history.objective == r0.history.objective && get_values(r.z) == get_values(r0.z)
      @test isfile(joinpath(out, "vtk", "optimization.pvd"))
    end
  end
  @testset "The README and the Getting started page show examples/mbb.jl" begin
    root = joinpath(@__DIR__, "..", "..")
    ex = read(joinpath(root, "examples", "mbb.jl"), String)
    body = strip(ex[findfirst("using Gridap", ex)[1]:findfirst("if abspath", ex)[1]-1])
    for doc in ("README.md", joinpath("docs", "src", "getting_started.md"))
      @test occursin(body, read(joinpath(root, doc), String))
    end
  end
end
