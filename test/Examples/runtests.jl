# The scripts in examples/ run here at a small size, so the documented scripts keep working.
include(joinpath(@__DIR__, "..", "..", "examples", "mbb.jl"))

@testset "Examples" begin
  @testset "MBB beam (examples/mbb.jl)" begin
    out = mktempdir()
    r = mbb(; nx = 30, ny = 10, maxiter = 12, out, verbose = 0)
    @test r.iterations == 12 && r.status === :maxiter
    @test r.history.objective[end] < r.history.objective[1]
    @test abs(r.history.constraints.volume[end]) < 1e-3
    @test isfile(joinpath(out, "checkpoint", "checkpoint.toml")) && isfile(joinpath(out, "vtk", "optimization.pvd"))
  end
end
