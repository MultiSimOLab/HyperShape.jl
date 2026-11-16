using Gridap
using Gridap.FESpaces
using Gridap.Geometry          # exports its own extend: extend_field must not clash
using HyperFEM
using HyperShape
using HyperShape.Variables: get_values
using HyperShape.Operators: num_solves
using HyperShape.PostProcess: _load
using Test

# times and files of a .pvd
function pvd_entries(path)
  ts, fs = Float64[], String[]
  for l in eachline(path)
    m = match(r"timestep=\"([^\"]+)\".*file=\"([^\"]+)\"", l)
    m === nothing && continue
    push!(ts, parse(Float64, m[1]))
    push!(fs, m[2])
  end
  ts, fs
end

# a recording driver
recorder(calls) = (post, ctx) -> push!(calls, (ctx.iteration, ctx.phase, post.iteration))

@testset "PostProcess" begin

  @testset "extend_field ($(dim)D)" for dim in (2, 3)
    n = dim == 2 ? (30, 10) : (6, 2, 2)
    model = CartesianDiscreteModel(dim == 2 ? (0, 3, 0, 1) : (0, 3, 0, 1, 0, 1), n)
    Ω = Triangulation(model)
    xc = [sum(c) / length(c) for c in get_cell_coordinates(Ω)]
    Ωd = Triangulation(model, [1.0 < x[1] < 2.0 for x in xc])        # design zone: the middle third
    Ωs = Triangulation(model, [x[1] > 0.5 for x in xc])              # larger than Ωd, not the whole model
    dΩ, dΩd, dΩs = Measure(Ω, 2), Measure(Ωd, 2), Measure(Ωs, 2)
    V0 = FESpace(Ωd, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
    V1 = FESpace(Ωd, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    g(x) = 0.2 + 0.6 * x[2]
    f0, f1 = interpolate_everywhere(g, V0), interpolate_everywhere(g, V1)
    proj(r) = tanh(4 * (r - 0.5))                     # a pointwise operation, like the projection of θ.ρ
    dir = mktempdir()
    for (name, f) in (("P0", f0), ("P1", f1), ("opP1", proj ∘ f1))
      fd = f(get_cell_points(dΩd))
      for (Ωt, dΩt, out) in ((Ω, dΩ, 1.0), (Ωs, dΩs, 0.25))
        e = extend_field(f, Ωt; outside = out)
        cmap = HyperShape.PostProcess._cell_map(Ωd, Ωt)
        vals = e(get_cell_points(dΩt))
        @test all(vals[i] == fd[cmap[i]] for i in eachindex(cmap) if cmap[i] > 0)           # bit for bit inside
        @test all(all(==(out), vals[i]) for i in eachindex(cmap) if cmap[i] == 0)            # exact outside
        @test sum(∫(e)dΩt) ≈ sum(∫(f)dΩd) + out * (sum(∫(1.0)dΩt) - sum(∫(1.0)dΩd)) rtol = 1e-12
        file = joinpath(dir, "e_$(dim)_$(name)_$(num_cells(Ωt))")
        HyperShape.PostProcess.WriteVTK.vtk_save(createvtk(Ωt, file; cellfields = ["rho" => e]))
        @test isfile(file * ".vtu")
      end
    end
    @test extend_field(f1, Ωd; outside = 1.0) === f1
    other = CartesianDiscreteModel(dim == 2 ? (0, 3, 0, 1) : (0, 3, 0, 1, 0, 1), n)
    @test_throws ArgumentError extend_field(f1, Triangulation(other); outside = 1.0)
  end

  @testset "Names" begin
    @test extend_field === HyperShape.PostProcess.extend_field
    @test get_pvd === HyperFEM.get_pvd
    @test all(n -> n in names(HyperShape), (:OptPostProcessor, :write_step, :extend_field, :get_pvd))
    @test_throws ArgumentError OptPostProcessor(recorder([]); every = 0)
    @test_throws ArgumentError OptPostProcessor(recorder([]); every = 1, name = "")
    q = compliance_problem()
    @test_throws ArgumentError solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 2, verbose = 0,
                                      post = ctx -> nothing)
    @test_throws ArgumentError get_pvd(OptPostProcessor(recorder([]); every = 1))
  end

  @testset "When the driver is called ($(design), $(nameof(typeof(opt))))" for design in (:P0, :P1), opt in (OC(), SiMPL())
    for (maxiter, initial, expected) in ((12, true, [0, 5, 10, 12]), (12, false, [5, 10, 12]), (10, true, [0, 5, 10]))
      q = compliance_problem(; design)
      metric = opt isa OC ? Euclidean() : LumpedMass(q.dΩ)
      calls = Tuple[]
      r = solve!(q.problem, SIMP(opt; metric); maxiter, verbose = 0, stop = nothing,
                 post = OptPostProcessor(recorder(calls); every = 5, initial))
      @test first.(calls) == expected
      @test all(c -> c[2] === :optimization && c[3] == c[1], calls)
      @test r.iterations == maxiter
    end
  end

  @testset "No extra solves; values at the accepted design" begin
    q1, q2 = compliance_problem(), compliance_problem()
    r1 = solve!(q1.problem, SIMP(OC(); metric = Euclidean()); maxiter = 6, verbose = 0, stop = nothing)
    designs = Vector{Float64}[]
    states = Vector{Float64}[]
    post = OptPostProcessor(every = 2) do post, ctx
      push!(designs, copy(get_free_dof_values(q2.x.ρ)))
      push!(states, copy(get_free_dof_values(q2.uh)))
    end
    r2 = solve!(q2.problem, SIMP(OC(); metric = Euclidean()); maxiter = 6, verbose = 0, stop = nothing, post)
    @test num_solves(q2.S) == num_solves(q1.S)
    @test sum(r2.history.forward) == sum(r1.history.forward) - 1     # the first forward solve is done before iteration 1
    @test r2.history.objective == r1.history.objective
    @test length(designs) == 4 && designs[end] == get_values(q2.z) && states[end] == get_free_dof_values(q2.uh)
  end

  @testset "Collection, files and history ($(design))" for design in (:P0, :P1)
    dir = joinpath(mktempdir(), "out")
    q = compliance_problem(; design)
    Ω = get_triangulation(q.uh)
    seen = Int[]
    post = OptPostProcessor(every = 5, filepath = dir) do post, ctx
      # the collection is on disk after every entry: readable during the run
      ctx.iteration > 0 && push!(seen, length(first(pvd_entries(joinpath(dir, "optimization.pvd")))))
      @test post.problem === q.problem
      write_step(post, Ω, "rho" => q.θ.ρ, "x" => q.x.ρ, "u" => q.uh)
    end
    r = solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 12, verbose = 0, stop = nothing, post)
    ts, fs = pvd_entries(joinpath(dir, "optimization.pvd"))
    @test ts == [0, 5, 10, 12] && seen == [1, 2, 3]
    @test all(f -> isfile(joinpath(dir, f)), fs) && fs[1] == "optimization_0000.vtu"
    lines = readlines(joinpath(dir, "history.csv"))
    @test length(lines) == 13
    @test parse.(Float64, [split(l, ",")[2] for l in lines[2:end]]) == r.history.objective
    # the same post-processor in a new run starts a new collection
    solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 3, verbose = 0, stop = nothing, post)
    @test first(pvd_entries(joinpath(dir, "optimization.pvd"))) == [0, 3]
    # other files and options: compress, a file written by the driver, several parts at one time
    dir2 = mktempdir()
    post2 = OptPostProcessor(every = 2, filepath = dir2, name = "run", history = false) do post, ctx
      write_step(post, Ω, "rho" => q.θ.ρ; compress = true)
      write_step(post, Ω, "u" => q.uh; name = "state")
      f = joinpath(post.filepath, "mine_$(ctx.iteration).vtu")
      write(f, "")
      get_pvd(post)[ctx.iteration] = f
    end
    solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 2, verbose = 0, stop = nothing, post = post2)
    ts, fs = pvd_entries(joinpath(dir2, "run.pvd"))
    @test ts == [0, 0, 0, 2, 2, 2] && fs[1:3] == ["run_0000.vtu", "state_0000.vtu", "mine_0.vtu"]
    @test occursin("part=\"2\"", read(joinpath(dir2, "run.pvd"), String))
    @test !isfile(joinpath(dir2, "history.csv"))
  end

  @testset "Built before or after the problem; without filepath" begin
    calls = Tuple[]
    post = OptPostProcessor(recorder(calls); every = 3)
    q = compliance_problem()
    solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 3, verbose = 0, stop = nothing, post)
    @test post.problem === q.problem && first.(calls) == [0, 3]
    @test_throws ArgumentError get_pvd(post)
    @test_throws ArgumentError write_step(post, get_triangulation(q.uh), "rho" => q.θ.ρ)
  end

  @testset "Resume: entries after the checkpoint dropped, the restart design not written again" begin
    dir, ck = mktempdir(), mktempdir()
    q = compliance_problem()
    calls = Tuple[]
    post = OptPostProcessor(every = 3, filepath = dir) do post, ctx
      push!(calls, (ctx.iteration,))
      write_step(post, get_triangulation(q.uh), "rho" => q.θ.ρ)
    end
    solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 10, verbose = 0, stop = nothing, post,
           checkpoint = Checkpoint(ck; every = 5))
    @test first.(calls) == [0, 3, 6, 9, 10]
    c = _load(joinpath(dir, "optimization.pvd"); upto = Inf)    # a later entry, as left by a longer run
    c[15] = joinpath(dir, "optimization_0010.vtu")
    empty!(calls)
    q2 = compliance_problem()
    post = OptPostProcessor(every = 3, filepath = dir) do p, ctx
      push!(calls, (ctx.iteration,))
      write_step(p, get_triangulation(q2.uh), "rho" => q2.θ.ρ)
    end
    solve!(q2.problem, SIMP(OC(); metric = Euclidean()); maxiter = 13, verbose = 0, stop = nothing, post,
           resume = load_checkpoint(ck))
    @test first.(calls) == [12, 13]
    @test first(pvd_entries(joinpath(dir, "optimization.pvd"))) == [0, 3, 6, 9, 10, 12, 13]
  end

  @testset "Stall and failure: the last accepted design is written" begin
    sign = Ref(1.0)
    q = compliance_problem(; sign)
    calls = Tuple[]
    flip = OptPostProcessor(every = 1, initial = false) do post, ctx
      push!(calls, (ctx.iteration,))
      ctx.iteration == 2 && (sign[] = -1.0)
    end
    r = solve!(q.problem, SIMP(SiMPL(max_backtracks = 1); metric = LumpedMass(q.dΩ)); maxiter = 10, verbose = 0,
               stop = nothing, post = flip)
    @test r.status === :stalled && r.iterations == 2 && first.(calls) == [1, 2]
    # failure: the forward problem fails at both attempts of iteration 2, and solves again at the last accepted design
    cnt = Ref(0)
    q = compliance_problem(; solve_flag = () -> (cnt[] += 1; !(cnt[] in (3, 4))))
    empty!(calls)
    post = OptPostProcessor(every = 5) do post, ctx
      push!(calls, (ctx.iteration,))
    end
    err = try
      solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 6, verbose = 0, stop = nothing,
             on_failure = RejectStep(shrink = 0.5, max = 2), post)
      nothing
    catch e
      e
    end
    @test err isa OptimizationFailure && err.iterations == 1
    @test first.(calls) == [0, 1]
    # a driver that throws after a failure does not hide the failure
    cnt[] = 0
    q = compliance_problem(; solve_flag = () -> (cnt[] += 1; !(cnt[] in (3, 4))))
    bad = OptPostProcessor(every = 5) do post, ctx
      ctx.iteration > 0 && error("driver")
    end
    io = IOBuffer()
    @test_throws OptimizationFailure solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 6, io, stop = nothing,
                                            on_failure = RejectStep(shrink = 0.5, max = 2), post = bad)
    @test occursin("post-processing of the last accepted design threw", String(take!(io)))
    # elsewhere, an error of the driver stops the optimisation
    @test_throws ErrorException solve!(compliance_problem().problem, SIMP(OC(); metric = Euclidean()); maxiter = 6,
                                       verbose = 0, stop = nothing, post = bad)
  end
  @testset "Review fixes" begin
    # vector and tensor fields, outside of the field type or a number per component
    model = CartesianDiscreteModel((0, 3, 0, 1), (12, 4))
    Ω = Triangulation(model)
    xc = [sum(c)[1] / length(c) for c in get_cell_coordinates(Ω)]
    Ωd = Triangulation(model, [1.0 < x < 2.0 for x in xc])
    Vv = FESpace(Ωd, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1); conformity = :H1)
    uv = interpolate_everywhere(x -> VectorValue(x[1], x[2]), Vv)
    dΩ, dΩd = Measure(Ω, 2), Measure(Ωd, 2)
    for (f, out, z) in ((uv, 0.0, VectorValue(0.0, 0.0)), (uv, VectorValue(1.0, 2.0), VectorValue(1.0, 2.0)),
                        (∇(uv), 0.0, TensorValue(0.0, 0.0, 0.0, 0.0)))
      e = extend_field(f, Ω; outside = out)
      cmap = HyperShape.PostProcess._cell_map(Ωd, Ω)
      vals = e(get_cell_points(dΩ))
      @test all(all(==(z), vals[i]) for i in eachindex(cmap) if cmap[i] == 0)
      file = joinpath(mktempdir(), "v")
      HyperShape.PostProcess.WriteVTK.vtk_save(createvtk(Ω, file; cellfields = ["f" => e]))
      @test isfile(file * ".vtu")
    end
    # names: a path separator, a one-argument driver
    @test_throws ArgumentError OptPostProcessor((p, c) -> nothing; every = 1, name = "a/b")
    @test_throws ArgumentError OptPostProcessor(ctx -> nothing; every = 1)
    # XML characters in file names, a WriteVTK-style line, the same file twice at one time
    dir = mktempdir()
    c = HyperShape.PostProcess._save(HyperShape.PostProcess.PVDCollection(joinpath(dir, "c.pvd")))
    c[0] = joinpath(dir, "a&b \"q\" <1>.vtu")
    @test_throws ArgumentError (c[0] = joinpath(dir, "a&b \"q\" <1>.vtu"))
    @test _load(joinpath(dir, "c.pvd"); upto = Inf).files == ["a&b \"q\" <1>.vtu"]
    write(joinpath(dir, "w.pvd"), """<?xml version="1.0"?>\n<VTKFile type="Collection">\n<Collection>\n""" *
          """<DataSet timestep="0.0" group="" part="0" file="w_0.vtu"/>\n<DataSet timestep="2.0" group="" part="0" file="w_2.vtu"/>\n""" *
          """</Collection>\n</VTKFile>\n""")
    w = _load(joinpath(dir, "w.pvd"); upto = 1)
    @test w.times == [0.0] && w.files == ["w_0.vtu"]
    # a relative filepath is fixed when the post-processor is built: a change of directory does not move the output
    root = mktempdir()
    post = cd(() -> OptPostProcessor((p, ctx) -> nothing; every = 1, filepath = "out"), root)
    @test post.filepath == joinpath(realpath(root), "out") || post.filepath == joinpath(root, "out")
    q = compliance_problem()
    calls = Int[]
    post = cd(root) do
      OptPostProcessor(every = 2, filepath = "rel") do p, ctx
        push!(calls, ctx.iteration)
        ctx.iteration == 0 && cd(mktempdir())          # the driver changes directory
        write_step(p, get_triangulation(q.uh), "rho" => q.θ.ρ)
      end
    end
    old = pwd()
    try
      solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 4, verbose = 0, stop = nothing, post)
    finally
      cd(old)
    end
    @test calls == [0, 2, 4] && first(pvd_entries(joinpath(post.filepath, "optimization.pvd"))) == [0, 2, 4]
    # a driver that throws does not lose the checkpoint of that iteration
    ck = mktempdir()
    bad = OptPostProcessor((p, ctx) -> (ctx.iteration == 4 && error("disk full")); every = 2)
    @test_throws ErrorException solve!(compliance_problem().problem, SIMP(OC(); metric = Euclidean()); maxiter = 4,
                                       verbose = 0, stop = nothing, post = bad, checkpoint = Checkpoint(ck; every = 2))
    @test load_checkpoint(ck).iteration == 4
    # a resumed run that starts with a continuation jump: no initial call, no extra forward solve
    nsolves = Ref(0)
    for withpost in (false, true)
      q = compliance_problem()
      ck = mktempdir()
      sched = Continuation(β = q.H => [1.0, 2.0, 4.0]; every = 4)
      solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 4, verbose = 0, stop = nothing, schedule = sched,
             checkpoint = Checkpoint(ck; every = 4))
      n0 = num_solves(q.S)
      calls = Int[]
      p = withpost ? OptPostProcessor((p, ctx) -> push!(calls, ctx.iteration); every = 3) : nothing
      solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 9, verbose = 0, stop = nothing, schedule = sched,
             resume = load_checkpoint(ck), post = p)
      withpost && @test calls == [6, 9]
      if withpost
        @test num_solves(q.S) - n0 == nsolves[]
      else
        nsolves[] = num_solves(q.S) - n0
      end
    end
    # the driver sees the accepted design with a nonlinear constraint and SiMPL's line search
    q = compliance_problem()
    Volθ = Volume(q.θ, q.dΩ; max_fraction = 0.4)
    ok = Bool[]
    post = OptPostProcessor(every = 1) do p, ctx
      push!(ok, get_free_dof_values(q.x.ρ) == get_values(q.z) == get_values(p.problem.z))
    end
    solve!(Optimization(q.z; objective = q.C, constraints = Volθ), SIMP(SiMPL(); metric = LumpedMass(q.dΩ));
           maxiter = 6, verbose = 0, stop = nothing, post)
    @test length(ok) == 7 && all(ok)
  end
end
