# The scripts in examples/ run here at a small size, so the documented scripts keep working.
include(joinpath(@__DIR__, "..", "..", "examples", "mbb.jl"))

@testset "Examples" begin
  @testset "MBB beam (examples/mbb.jl)" begin
    out = mktempdir()
    r = run_mbb(; nx = 30, ny = 10, maxiter = 12, out, verbose = 0)
    @test r.iterations == 12 && r.status === :maxiter
    @test r.history.objective[end] < r.history.objective[1]
    @test abs(r.history.constraints.volume[end]) < 1e-3
    @test isfile(joinpath(out, "checkpoint", "checkpoint.toml")) && isfile(joinpath(out, "vtk", "optimization.pvd"))
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
