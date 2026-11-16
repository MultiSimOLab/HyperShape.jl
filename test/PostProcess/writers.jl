using Gridap
using Gridap.FESpaces
using HyperFEM
using HyperShape
using HDF5
using Test
using HyperShape.PostProcess: _plan, _lazy_plan, _values!, _ExtNode, _LazyNode, npoints, ncells

# fast values of a field against Gridap's evaluation at the vertices (averaged at the nodes: exact for continuous fields)
function fast_vs_gridap(w, f)
  node = _plan(w, f)
  node isa _LazyNode && return false
  v = collect(_values!(node))
  node.loc === :cell && return v == [first(c) for c in f(Gridap.CellData.CellPoint(Gridap.Arrays.Fill(w.mesh.verts, ncells(w.mesh)), w.Ω, Gridap.CellData.ReferenceDomain()))]
  ref = collect(_values!(_lazy_plan(w, f)))
  isapprox(v, ref; rtol = 1e-12, atol = 1e-12)
end

vtu_npoints(file) = parse(Int, match(r"NumberOfPoints=\"(\d+)\"", read(file, String))[1])

@testset "FieldWriters" begin

  @testset "Values from the dofs ($(name))" for (name, model) in (
      ("QUAD", CartesianDiscreteModel((0, 2, 0, 1), (6, 3))),
      ("HEX", CartesianDiscreteModel((0, 2, 0, 1, 0, 1), (3, 2, 2))),
      ("TRI", simplexify(CartesianDiscreteModel((0, 2, 0, 1), (4, 3)))),
      ("TET", simplexify(CartesianDiscreteModel((0, 2, 0, 1, 0, 1), (2, 2, 2)))))
    D = num_cell_dims(model)
    labels = get_face_labeling(model)
    Ω = Triangulation(model)
    g(x) = 0.3 + x[1]^2 - 0.5 * x[2]
    gv(x) = D == 2 ? VectorValue(x[1], -x[2]) : VectorValue(x[1], -x[2], x[3])
    V0 = FESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
    V1 = FESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1, dirichlet_tags = "boundary")
    U1 = TrialFESpace(V1, x -> 2.0)                        # Dirichlet values differ from the free ones
    Vv = FESpace(model, ReferenceFE(lagrangian, VectorValue{D,Float64}, 1); conformity = :H1)
    V2 = FESpace(model, ReferenceFE(lagrangian, VectorValue{D,Float64}, 2); conformity = :H1)
    f0 = interpolate_everywhere(g, V0)
    f1 = interpolate(g, U1)
    fv = interpolate_everywhere(gv, Vv)
    f2 = interpolate_everywhere(gv, V2)
    dir = mktempdir()
    w = FieldWriter(Ω, joinpath(dir, "f"); async = false)
    @test npoints(w.mesh) == num_vertices(model) && ncells(w.mesh) == num_cells(model)
    for f in (f0, f1, fv, f2, (r -> tanh(4 * (r - 0.5))) ∘ f1)
      @test fast_vs_gridap(w, f)
    end
    @test fast_vs_gridap(w, Operation(+)(f1, f1))
    write_fields(w, 0, "p0" => f0, "p1" => f1, "v" => fv, "q2" => f2)
    close(w)
    @test vtu_npoints(joinpath(dir, "f_0000.vtu")) == num_vertices(model)
  end

  @testset "Partial triangulations, extend_field, slow path" begin
    model = CartesianDiscreteModel((0, 3, 0, 1), (12, 4))
    Ω = Triangulation(model)
    xc = [sum(c)[1] / length(c) for c in get_cell_coordinates(Ω)]
    Ωs = Triangulation(model, xc .> 0.5)
    Ωd = Triangulation(model, 1.0 .< xc .< 2.0)
    V1 = FESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    f = interpolate_everywhere(x -> x[1] + x[2], V1)
    Vd = FESpace(Ωd, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    Vd0 = FESpace(Ωd, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
    fd, fd0 = interpolate_everywhere(x -> x[1], Vd), interpolate_everywhere(x -> x[1], Vd0)
    dir = mktempdir()
    ws = FieldWriter(Ωs, joinpath(dir, "s"); async = false)
    @test ncells(ws.mesh) == num_cells(Ωs) && fast_vs_gridap(ws, f)        # a field of the whole model, on a part
    @test_throws ArgumentError write_fields(ws, 0, "fd" => fd)            # a field of a smaller part
    w = FieldWriter(Ω, joinpath(dir, "w"); async = false)
    for (h, out) in ((fd, 1.0), (fd0, 0.5))
      e = extend_field(h, Ω; outside = out)
      node = _plan(w, e)
      @test node isa _ExtNode
      v = collect(_values!(node))
      @test count(==(out), v) > 0 && all(x -> x == out || 1.0 <= x <= 2.0, v)
    end
    @test sum(∫(extend_field(fd0, Ω; outside = 0.5))Measure(Ω, 2)) ≈ sum(∫(fd0)Measure(Ωd, 2)) + 0.5 * 2.0
    g = ∇(f)                                                              # no fast path: Gridap at the vertices
    @test_logs (:warn, r"evaluated by Gridap") write_fields(w, 0, "grad" => g)
    @test _plan(w, f) isa HyperShape.PostProcess._Leaf
    @test w.plans["grad"][2] isa _LazyNode
    close(w)
    @test close(w) === w                                                   # idempotent
    @test_throws ArgumentError write_fields(w, 1, "f" => f)
  end

  @testset "VTKHDF, Float32, async, resume" begin
    model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (4, 3, 2))
    Ω = Triangulation(model)
    V0 = FESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
    Vv = FESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1); conformity = :H1)
    f0 = interpolate_everywhere(x -> x[1], V0)
    fv = interpolate_everywhere(x -> VectorValue(x[1], x[2], x[3]), Vv)
    dir = mktempdir()
    for async in (false, true)
      w = FieldWriter(Ω, joinpath(dir, "h$async"); format = :vtkhdf, precision = Float32, async)
      for t in 0:3
        write_fields(w, t, "rho" => f0, "u" => fv)
      end
      close(w)
      h5open(joinpath(dir, "h$async.vtkhdf")) do h
        g = h["VTKHDF"]
        @test read(attributes(g["Steps"])["NSteps"]) == 4 && read(g["Steps/Values"]) == [0.0, 1.0, 2.0, 3.0]
        @test eltype(g["Points"]) == Float64 && eltype(g["CellData/rho"]) == Float32
        @test size(g["PointData/u"]) == (3, 4 * npoints(w.mesh)) && length(g["CellData/rho"]) == 4 * ncells(w.mesh)
      end
    end
    data(file) = h5open(h -> Dict(k => read(h["VTKHDF/" * k]) for k in ("Points", "Connectivity", "CellData/rho", "PointData/u", "Steps/Values")), file)
    @test data(joinpath(dir, "hfalse.vtkhdf")) == data(joinpath(dir, "htrue.vtkhdf"))   # async writes the same data
    # resume: steps up to 1 kept, then appended
    w = FieldWriter(Ω, joinpath(dir, "hfalse"); format = :vtkhdf, precision = Float32, async = false, resume_upto = 1)
    write_fields(w, 2, "rho" => f0, "u" => fv)
    close(w)
    h5open(joinpath(dir, "hfalse.vtkhdf")) do h
      @test read(h["VTKHDF/Steps/Values"]) == [0.0, 1.0, 2.0]
      @test length(h["VTKHDF/CellData/rho"]) == 3 * num_cells(model)
    end
    # VTU: async and sync give the same files
    for async in (false, true)
      w = FieldWriter(Ω, joinpath(dir, "v$async"); async)
      for t in 0:2
        write_fields(w, t, "rho" => f0, "u" => fv)
      end
      close(w)
    end
    @test all(read(joinpath(dir, "vfalse_000$i.vtu")) == read(joinpath(dir, "vtrue_000$i.vtu")) for i in 0:2)
    # an error of an asynchronous write reaches the caller
    sub = joinpath(dir, "gone")
    w = FieldWriter(Ω, joinpath(sub, "x"); async = true)
    rm(sub; recursive = true)
    write_fields(w, 0, "rho" => f0)
    @test_throws Exception close(w)
    @test_throws ArgumentError FieldWriter(Ω, joinpath(dir, "bad"); format = :xdmf)
  end

  @testset "write_step with the writers of the post-processor ($(format))" for format in (:vtu, :vtkhdf)
    q = compliance_problem()
    Ω = get_triangulation(q.uh)
    dir = mktempdir()
    post = OptPostProcessor(every = 2, filepath = dir, format = format) do p, ctx
      write_step(p, Ω, "rho" => q.θ.ρ, "x" => q.x.ρ, "u" => q.uh)
    end
    solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 5, verbose = 0, stop = nothing, post)
    @test isempty(post.writers)                                        # closed by solve!
    if format === :vtu
      @test first(pvd_entries(joinpath(dir, "optimization.pvd"))) == [0, 2, 4, 5]
    else
      @test !isfile(joinpath(dir, "optimization.pvd"))
      @test h5open(h -> read(h["VTKHDF/Steps/Values"]), joinpath(dir, "optimization.vtkhdf")) == [0.0, 2.0, 4.0, 5.0]
      # resume from a checkpoint at 4: the steps after it are dropped
      ck = mktempdir()
      q = compliance_problem()
      solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 4, verbose = 0, stop = nothing, checkpoint = Checkpoint(ck; every = 2))
      Ω2 = get_triangulation(q.uh)
      post2 = OptPostProcessor(every = 3, filepath = dir, format = :vtkhdf) do p, ctx
        write_step(p, Ω2, "rho" => q.θ.ρ, "x" => q.x.ρ, "u" => q.uh)
      end
      solve!(q.problem, SIMP(OC(); metric = Euclidean()); maxiter = 7, verbose = 0, stop = nothing, post = post2,
             resume = load_checkpoint(ck))
      @test h5open(h -> read(h["VTKHDF/Steps/Values"]), joinpath(dir, "optimization.vtkhdf")) == [0.0, 2.0, 4.0, 6.0, 7.0]
    end
    @test_throws ArgumentError OptPostProcessor((p, c) -> nothing; every = 1, format = :xdmf)
  end
end
