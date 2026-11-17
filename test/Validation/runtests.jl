# Fast validation cases (the long ones are in validation/, outside CI): linear elasticity, against top88 with the PDE
# filter and a plain-Julia port of it; Taylor tests of elasticity in 2D and 3D; a small 3D cantilever.
using Test
include(joinpath(@__DIR__, "..", "..", "validation", "cases.jl"))

@testset "Validation" begin

  @testset "top88 port (reference)" begin
    res = top88pde(60, 20, 0.5, 2.4)
    @test res.c ≈ 237.5996 rtol = 1e-5                     # the last compliance top88 prints
    @test res.loop == 140
  end

  base = (; case = :top88, nx = 60, ny = 20, vf = 0.5, rmin = 2.4, Emin = 1e-9)
  t88 = (; opt = OC(adapt = nothing), stop = StopCriteria(change = 0.01))

  @testset "MBB 60×20 against top88 with the PDE filter (237.60)" begin
    r = mbb(; base..., t88..., cellmean = true)              # the formulation of top88: cell mean of the filtered density
    @test r.result.status === :converged && r.result.iterations <= 150
    @test r.result.objective ≈ 237.60 rtol = 1e-3
    @test sum(r.ρ) / length(r.ρ) ≈ 0.5 atol = 1e-4
    # the same design evaluated with the finite elements of top88
    @test compliance(T88(60, 20, 2.4), gridap_to_top88(r.ρ, 60, 20)) ≈ 237.60 rtol = 1e-3
    for order in (0, 1)                                       # density at the Gauss points (the usual use of HyperShape)
      g = mbb(; base..., t88..., order)
      @test g.result.status === :converged
      @test g.result.objective ≈ 237.60 rtol = 2e-2
    end
  end

  @testset "Default optimisers ($(nameof(typeof(opt))), P$order)" for opt in (OC(), SiMPL()), order in (0, 1)
    r = mbb(; base..., opt, order, maxiter = 400)
    @test r.result.status === :converged
    @test r.result.objective ≈ 237.60 rtol = 2e-2
  end

  @testset "Taylor tests of elasticity (P$order)" for order in (0, 1)
    @test mbb(; base..., nx = 12, ny = 4, rmin = 1.5, order, check = true).passed
    @test mbb(; base..., nx = 12, ny = 4, rmin = 1.5, order, check = true, cellmean = true).passed
    @test cantilever3d(; n = (8, 4, 4), order, check = true).passed
  end

  @testset "Filter and cell mean against the port of top88" begin
    model = CartesianDiscreteModel((0, 60, 0, 20), (60, 20))
    dΩ = Measure(Triangulation(model), 2)
    Vρ = design_space(model, 0)
    ρ0 = [0.2 + 0.6 * abs(sin(1.7i)) for i in 1:num_free_dofs(Vρ)]
    z, x, θ = single_field_design(:ρ, Vρ; bounds = (0.0, 1.0), init = ρ0, physical = density_chain(model, dΩ, 2.4; cellmean = true))
    @test gridap_to_top88(cell_means(θ, dΩ, 1.0), 60, 20) ≈ filt(T88(60, 20, 2.4), gridap_to_top88(ρ0, 60, 20)) atol = 1e-12
  end

  @testset "Taylor test of the SiMPL setup (plane strain, body force, partial roller)" begin
    @test mbb(; case = :simpl, nx = 48, ny = 16, vf = 0.3, rmin = 0.1, Emin = 1e-6, check = true).passed
  end

  @testset "3D cantilever 8×4×4" begin
    r = cantilever3d(; n = (8, 4, 4), opt = SiMPL(), maxiter = 10, stop = nothing)
    h = r.result.history
    @test r.result.iterations == 10 && all(i -> h.objective[i+1] <= h.objective[i] * (1 + 1e-12), 1:9)
    @test all(<=(1e-6), h.constraint) && r.asymmetry < 1e-8
  end
end
