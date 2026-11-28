# ReactionDiffusion(…; phi_bc): values of the design field held by the step (:zero_boundary, "tag" => value). The
# nodes held, one step against the algebra by hand, the values kept at every iteration, the exact volume with values
# held at a bound, the frozen zone over a tag, resume, allocations and errors. Uses the helpers of
# reaction_diffusion.jl (rd_cantilever, rd, rd_start, VT, ZTrack).
using Test
using LinearAlgebra
using SparseArrays
using Gridap
using Gridap.FESpaces
using HyperFEM
using HyperShape

# the coordinates of the free values of a P1 space
free_coords(V) = (get_free_dof_values(interpolate(x -> x[1], V)), get_free_dof_values(interpolate(x -> x[2], V)))
on_box(x, y, (x0, x1, y0, y1)) = abs(x - x0) < 1e-9 || abs(x - x1) < 1e-9 || abs(y - y0) < 1e-9 || abs(y - y1) < 1e-9

@testset "ReactionDiffusion phi_bc" begin

  @testset "1. the nodes held: boundary of quadrilaterals, triangles, a subdomain; tags; frozen zones win" begin
    for simplex in (false, true)
      c = rd_cantilever(; nx = 20, ny = 10, frozen = false, simplex)
      U = c.z.blocks[1].space
      xs, ys = free_coords(c.Vφ)
      p = RDM._phi_bc_nodes(:zero_boundary, U, -1.0, 1.0)
      @test p.idx == findall(i -> on_box(xs[i], ys[i], (0, 2, 0, 1)), eachindex(xs))
      @test length(p.idx) == 2 * (20 + 10) && all(iszero, p.val) && p.nfrozen == 0
      q = RDM._phi_bc_nodes(RDM._phi_bc_arg("load" => 0.5), U, -1.0, 1.0)
      @test q.idx == findall(i -> xs[i] > 2 - 1e-9 && abs(ys[i] - 0.5) <= 0.1 + 1e-9, eachindex(xs)) && all(==(0.5), q.val)
      # two tags; a node shared by two tags with the same value
      r = RDM._phi_bc_nodes(RDM._phi_bc_arg(("load" => 0.5, "clamp" => 0.5)), U, -1.0, 1.0)
      @test sort(r.idx) == sort(union(q.idx, findall(<(1e-9), xs)))
    end
    # a design field on a subdomain (x < 1): its own boundary, x = 1 included
    model = CartesianDiscreteModel((0, 2, 0, 1), (20, 10))
    cc = vec([sum(x[1] for x in c) / 4 for c in get_cell_coordinates(Triangulation(model))])
    Ω1 = Triangulation(model, findall(<(1), cc))
    V1 = TestFESpace(Ω1, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    xs, ys = free_coords(V1)
    p = RDM._phi_bc_nodes(:zero_boundary, V1, -1.0, 1.0)
    @test p.idx == findall(i -> on_box(xs[i], ys[i], (0, 1, 0, 1)), eachindex(xs)) && length(p.idx) == 2 * (10 + 10)
    # the frozen zone around the load wins: its nodes are not values of z (counted), the tag "load" holds none
    c = rd_cantilever(; nx = 20, ny = 10)
    U = c.z.blocks[1].space
    xs, ys = free_coords(c.Vφ)
    p = RDM._phi_bc_nodes(:zero_boundary, U, -1.0, 1.0)
    nb = 2 * (20 + 10)
    @test p.idx == findall(i -> on_box(xs[i], ys[i], (0, 2, 0, 1)), eachindex(xs))
    @test p.nfrozen > 0 && length(p.idx) + p.nfrozen == nb
    q = RDM._phi_bc_nodes(RDM._phi_bc_arg("load" => 1.0), U, -1.0, 1.0)
    @test isempty(q.idx) && q.nfrozen == 3          # the nodes of "load" (y = 0.4, 0.5, 0.6), all frozen
  end

  @testset "2. one step against the algebra by hand: the reduced system ($mass mass)" for mass in (:lumped, :consistent)
    c = rd_cantilever(; nx = 20, ny = 10, design = :density, frozen = false)
    m = rd(; mass, phi_bc = ("load" => 1.0, "clamp" => 0.25))
    RDM.check_problem(m, c.problem)
    RDM.start_design!(m, c.problem)
    fo, z, s = rd_start(c, m)
    pin = s.pin
    @test !isempty(pin) && !s.unit && s.φc[pin] == zeros(length(pin)) && minimum(s.φc[setdiff(eachindex(z), pin)]) > 0
    z0 = copy(z)
    @test pin == m.step.pins[].idx && z0[pin] == m.step.pins[].val && get_values(c.z)[pin] == m.step.pins[].val
    info = OPT.step!(m.step, s, z, fo, c.problem.ctx)
    OPT.objective(fo, z0)
    g = OPT.gradient!(similar(z0), fo, z0)
    Vt = c.Vφ
    dΩ = Measure(Triangulation(c.model), 2)
    K = assemble_matrix((u, v) -> ∫(∇(u) ⋅ ∇(v))dΩ, Vt, Vt)
    mv = assemble_vector(v -> ∫(v)dΩ, Vt)
    Mc = assemble_matrix((u, v) -> ∫(u * v)dΩ, Vt, Vt)
    area = sum(∫(1.0)dΩ)
    τℓ² = 2e-4 * area
    A = Matrix((mass === :lumped ? Diagonal(mv) : Mc) / 0.4 + τℓ² * K)
    F = setdiff(eachindex(z0), pin)
    C = area / sum(abs, g)
    r = -τℓ² .* (K * z0) .- C .* g
    δ = zeros(length(z0)); δ[F] = A[F, F] \ r[F]
    φc = zeros(length(z0)); φc[F] = A[F, F] \ mv[F]
    @test s.φc ≈ φc rtol = 1e-12
    λ = info.multipliers[1]
    @test z ≈ clamp.(z0 .+ δ .- λ .* φc, 0.0, 1.0) atol = 1e-12
    @test z[pin] == z0[pin]
    @test abs(OPT.constraint(fo, 1, z)) <= VT
  end

  @testset "3. :zero_boundary: the boundary stays at 0 at every iteration; the volume is met ($(nameof(typeof(iface))))" for iface in (Ersatz(), Step())
    c = rd_cantilever(; nx = 40, ny = 20, interface = iface)
    U = c.z.blocks[1].space
    xs, ys = free_coords(c.Vφ)
    bnd = findall(i -> on_box(xs[i], ys[i], (0, 2, 0, 1)), eachindex(xs))
    tr = ZTrack(c.z, Vector{Float64}[])
    io = IOBuffer()
    r = solve!(c.problem, rd(; phi_bc = :zero_boundary); maxiter = 40, stop = nothing, verbose = 1, io, monitors = (zt = tr,))
    out = String(take!(io))
    @test occursin("phi_bc = :zero_boundary: $(length(bnd)) nodal values held at 0 on the boundary of the design domain", out)
    @test occursin("nodes there are frozen values of a zone (the zone wins)", out)
    @test length(tr.zs) == 40 && all(zk -> all(iszero, zk[bnd]), tr.zs)
    @test all(isfinite, r.history.objective) && abs(r.history.constraints.volume[end]) <= VT
    @test get_dirichlet_dof_values(U) == ones(num_dirichlet_dofs(U))       # the frozen zone kept
  end

  @testset "4. values held at the upper bound (φ̌ = 0 there): exact volume, no NaN ($design, $(nameof(typeof(iface))))" for
      (design, iface) in ((:density, Ersatz()), (:levelset, Ersatz()), (:levelset, Step()))
    c = rd_cantilever(; nx = 40, ny = 20, design, interface = iface, frozen = false)
    xs, ys = free_coords(c.Vφ)
    held = findall(i -> xs[i] > 2 - 1e-9 && abs(ys[i] - 0.5) <= 0.1 + 1e-9, eachindex(xs))
    tr = ZTrack(c.z, Vector{Float64}[])
    r = solve!(c.problem, rd(; phi_bc = ("load" => 1.0,)); maxiter = 30, stop = nothing, verbose = 0, monitors = (zt = tr,))
    @test all(zk -> all(==(1.0), zk[held]), tr.zs)
    @test all(isfinite, r.history.objective) && all(isfinite, r.history.constraints.volume)
    @test all(isfinite, r.history.multipliers.volume)
    vols = r.history.constraints.volume
    design === :density ? (@test all(v -> abs(v) <= VT, vols)) : (@test vols[end] <= VT)    # Step(): the feasible side
  end

  @testset "5. :neumann (the default) is the run of before: no pins, φ̌ = Δt·1 without a solve" begin
    c = rd_cantilever(; nx = 20, ny = 10, frozen = false)
    fo, z, s = rd_start(c, rd())
    @test isempty(s.pin) && isempty(s.pinned) && isempty(s.mp) && s.unit && s.nsolves == 0
    a, b = rd_cantilever(; nx = 30, ny = 15), rd_cantilever(; nx = 30, ny = 15)
    ra = solve!(a.problem, rd(); maxiter = 10, stop = nothing, verbose = 0)
    rb = solve!(b.problem, rd(; phi_bc = :neumann); maxiter = 10, stop = nothing, verbose = 0)
    @test get_values(ra.z) == get_values(rb.z) && ra.history.objective == rb.history.objective
  end

  @testset "6. written before the first evaluation: the scale :initial is taken at the held design" begin
    c = rd_cantilever(; nx = 30, ny = 15, design = :density, frozen = false)
    solve!(c.problem, rd(; phi_bc = ("load" => 0.0,)); maxiter = 1, stop = nothing, verbose = 0)
    s1 = get_scale(c.C)
    d = rd_cantilever(; nx = 30, ny = 15, design = :density, frozen = false)
    xs, ys = free_coords(d.Vφ)
    v = copy(get_values(d.z))
    v[findall(i -> xs[i] > 2 - 1e-9 && abs(ys[i] - 0.5) <= 0.1 + 1e-9, eachindex(xs))] .= 0.0
    set_values!(d.z, v)
    solve!(d.problem, rd(); maxiter = 1, stop = nothing, verbose = 0)
    @test s1 == get_scale(d.C)
  end

  @testset "7. checkpoint and resume bit for bit with held values" begin
    m() = rd(; phi_bc = :zero_boundary)
    ref = rd_cantilever(; nx = 30, ny = 15)
    r20 = solve!(ref.problem, m(); maxiter = 20, stop = nothing, verbose = 0)
    dir = mktempdir()
    a = rd_cantilever(; nx = 30, ny = 15)
    solve!(a.problem, m(); maxiter = 10, stop = nothing, verbose = 0, checkpoint = Checkpoint(filepath = dir, every = 10))
    b = rd_cantilever(; nx = 30, ny = 15)
    rb = solve!(b.problem, m(); maxiter = 20, stop = nothing, verbose = 0, resume = dir)
    @test get_values(rb.z) == get_values(r20.z) && rb.history.objective == r20.history.objective
  end

  @testset "8. no allocations in the pieces of the step that see the pins" begin
    c = rd_cantilever(; nx = 20, ny = 10, design = :density, frozen = false)
    m = rd(; phi_bc = ("load" => 1.0,))
    RDM.check_problem(m, c.problem)
    fo, z, s = rd_start(c, m)
    RDM._zero_pins!(s.r, s.pin)
    @test (@allocated RDM._zero_pins!(s.r, s.pin)) == 0
    RDM._free_minimum(s.φc, s.pinned)
    @test (@allocated RDM._free_minimum(s.φc, s.pinned)) == 0
    OPT.step!(m.step, s, z, fo, c.problem.ctx)
    lb, ub = OPT.bounds(fo)
    RDM._multiplier!(m.step, s, fo, 0.0, lb, ub)
    @test (@allocated RDM._multiplier!(m.step, s, fo, 0.0, lb, ub)) == 0
    RDM._rebuild!(m.step, s)
    @test s.φc[s.pin] == zeros(length(s.pin))
  end

  @testset "9. errors" begin
    @test_throws r"phi_bc = \(\"load\" => 1,\) is not offered with DoubleWell" ReactionDiffusion(; κ = 1e-5, reaction = DoubleWell(), phi_bc = ("load" => 1.0,))
    @test_throws r"phi_bc = :zero_boundary is not offered with DoubleWell" ReactionDiffusion(; κ = 1e-5,
      reaction = DoubleWell(scheme = :explicit, substeps = 20), phi_bc = :zero_boundary)
    @test_throws r"phi_bc must be :neumann, :zero_boundary or" rd(; phi_bc = :dirichlet)
    @test_throws r"phi_bc must be :neumann, :zero_boundary or" rd(; phi_bc = ())
    @test_throws r"phi_bc must be :neumann, :zero_boundary or" rd(; phi_bc = ("load" => "one",))
    @test_throws r"phi_bc must be :neumann, :zero_boundary or" rd(; phi_bc = (:load => 1.0,))
    c = rd_cantilever(; nx = 10, ny = 5, frozen = false)
    @test_throws r"\"lod\" is not a tag of the model" solve!(c.problem, rd(; phi_bc = ("lod" => 1.0,)); maxiter = 1, verbose = 0)
    @test_throws r"the value 2 of \"load\" is outside the bounds \[-1, 1\]" solve!(c.problem, rd(; phi_bc = "load" => 2.0); maxiter = 1, verbose = 0)
    @test_throws r"two values to a node shared by two tags" solve!(c.problem, rd(; phi_bc = ("load" => 1.0, "boundary" => 0.0)); maxiter = 1, verbose = 0)
    @test_throws r"gives a tag more than once: load" rd(; phi_bc = ("load" => 1.0, "load" => 0.5))
    c1 = rd_cantilever(; nx = 4, ny = 1, frozen = false)                     # every node is on the boundary
    @test_throws r"holds every value of the design field" solve!(c1.problem, rd(; phi_bc = :zero_boundary); maxiter = 1, verbose = 0)
    # the volume out of reach because of the held values: the message says so
    cv = rd_cantilever(; nx = 10, ny = 5, design = :density, fraction = 0.05, frozen = false)
    @test_throws r"the 30 values held by phi_bc count as they are" solve!(cv.problem, rd(; phi_bc = ("boundary" => 1.0,)); maxiter = 3, stop = nothing, verbose = 0)
    # a conflict only on nodes of z: the nodes of "load" are frozen (the zone wins), no conflict with "boundary"
    cf = rd_cantilever(; nx = 20, ny = 10)
    p = RDM._phi_bc_nodes(RDM._phi_bc_arg(("load" => 1.0, "boundary" => 0.0)), cf.z.blocks[1].space, -1.0, 1.0)
    @test all(iszero, p.val) && p.pertag == [0, length(p.idx)] && p.nfrozen > 0
    cd = rd_cantilever(; nx = 10, ny = 5, design = :density, frozen = false, bounds = (0.1, 1.0))
    @test_throws r":zero_boundary holds 0, outside the bounds" solve!(cd.problem, rd(; phi_bc = :zero_boundary); maxiter = 1, verbose = 0)
    # a tag with no node in the design domain (a subdomain x < 1 and the load at x = 2)
    model = CartesianDiscreteModel((0, 2, 0, 1), (8, 4))
    add_tag_from_vertex_filter!(get_face_labeling(model), "right", model, x -> x[1] > 2 - 1e-9)
    cc = vec([sum(x[1] for x in c) / 4 for c in get_cell_coordinates(Triangulation(model))])
    V1 = TestFESpace(Triangulation(model, findall(<(1), cc)), ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    @test_throws r"the tag \"right\" has no node in the design domain" RDM._phi_bc_nodes(RDM._phi_bc_arg("right" => 0.0), V1, -1.0, 1.0)
    add_tag_from_vertex_filter!(get_face_labeling(model), "left", model, x -> x[1] < 1e-9)
    # one tag without nodes among several is an error too
    @test_throws r"the tag \"right\" has no node in the design domain" RDM._phi_bc_nodes(RDM._phi_bc_arg(("left" => 0.0, "right" => 0.5)), V1, -1.0, 1.0)
    p = RDM._phi_bc_nodes(RDM._phi_bc_arg(("left" => 0.0,)), V1, -1.0, 1.0)
    @test length(p.idx) == 5 && p.pertag == [5]
  end

  @testset "10. non-regression: the default against values recorded on main (1dd327a)" begin
    # history.objective of 8 iterations recorded on main before phi_bc (Mac M5 Max, Julia 1.13); bit for bit there, a
    # tolerance of 1e-10 for other platforms (the rounding of the factorisations)
    ref = Dict(
      :ersatz => [1.0004034627363658, 1.0030287268344609, 1.010190787804603, 1.0187091443000327, 1.0313625914309277,
                  1.0473230973802459, 1.0593891531075905, 1.0768994220311674],
      :step => [1.0004166525136642, 1.0038715845064932, 1.0102912976191054, 1.0196452152808795, 1.0326156570857479,
                1.0440563460480992, 1.061780178315566, 1.0863697580876612],
      :density => [0.8023894651039728, 0.9193038673657785, 0.5997300407187898, 0.4073788395129964, 0.335571839067122,
                   0.2936186852327044, 0.27901588496348534, 0.2728620606935083])
    for (k, kw) in ((:ersatz, (;)), (:step, (; interface = Step())), (:density, (; design = :density)))
      c = rd_cantilever(; nx = 20, ny = 10, kw...)
      r = solve!(c.problem, rd(); maxiter = 8, stop = nothing, verbose = 0)
      @test r.history.objective ≈ ref[k] rtol = 1e-10
    end
  end

  @testset "11. a rebuild (RebuildCheck) with :zero_boundary is the run without it, bit for bit" begin
    p1, p2 = rd_factories(), rd_factories()
    f1 = solve!(p1, rd(; phi_bc = :zero_boundary); maxiter = 15, stop = nothing, verbose = 0)
    f2 = solve!(p2, rd(; phi_bc = :zero_boundary); maxiter = 15, stop = nothing, verbose = 0,
                discretization = RDM.RebuildCheck(every = 6))
    @test get_values(f1.z) == get_values(f2.z) && f1.history.objective == f2.history.objective
    V = RDM._test_space(f2.z.blocks[1].space)
    xs, ys = free_coords(V)
    @test all(iszero, get_values(f2.z)[findall(i -> on_box(xs[i], ys[i], (0, 2, 0, 1)), eachindex(xs))])
  end

  @testset "12. triangles: a full run with :zero_boundary ($(nameof(typeof(iface))))" for iface in (Ersatz(), Step())
    c = rd_cantilever(; nx = 20, ny = 10, simplex = true, interface = iface, frozen = false)
    xs, ys = free_coords(c.Vφ)
    bnd = findall(i -> on_box(xs[i], ys[i], (0, 2, 0, 1)), eachindex(xs))
    tr = ZTrack(c.z, Vector{Float64}[])
    r = solve!(c.problem, rd(; phi_bc = :zero_boundary); maxiter = 25, stop = nothing, verbose = 0, monitors = (zt = tr,))
    @test all(isfinite, r.history.objective) && all(zk -> all(iszero, zk[bnd]), tr.zs)
    @test r.history.constraints.volume[end] <= VT
  end

  @testset "13. a rejected step and a continuation of τ and Δt with held values" begin
    c = rd_cantilever(; nx = 20, ny = 10, frozen = false, solve_flag = ctx -> !(ctx.iteration == 3 && ctx.attempt == 1))
    xs, ys = free_coords(c.Vφ)
    bnd = findall(i -> on_box(xs[i], ys[i], (0, 2, 0, 1)), eachindex(xs))
    tr = ZTrack(c.z, Vector{Float64}[])
    r = solve!(c.problem, rd(; phi_bc = :zero_boundary); maxiter = 6, stop = nothing, verbose = 0, monitors = (zt = tr,))
    @test r.history.attempts[3] == 2 && r.state.nfactor == 3                 # halved, then back
    @test all(zk -> all(iszero, zk[bnd]), tr.zs) && all(iszero, r.state.φc[r.state.pin])
    pτ, pΔt = Parameter(2e-4), Parameter(0.1)
    c = rd_cantilever(; nx = 20, ny = 10, frozen = false)
    tr = ZTrack(c.z, Vector{Float64}[])
    r = solve!(c.problem, ReactionDiffusion(; τ = pτ, Δt = pΔt, phi_bc = :zero_boundary);
               maxiter = 6, stop = nothing, verbose = 0, monitors = (zt = tr,),
               continuation = Continuation(τ = pτ => [2e-4, 1e-3], Δt = pΔt => [0.1, 0.05]; every = 3))
    @test r.state.nfactor == 2 && all(zk -> all(iszero, zk[bnd]), tr.zs) && all(isfinite, r.history.objective)
  end

  @testset "14. a solve of the user, and volume = nothing (the keywords of Kim et al. 2020), with held values" begin
    a, b = rd_cantilever(; nx = 30, ny = 15, frozen = false), rd_cantilever(; nx = 30, ny = 15, frozen = false)
    ra = solve!(a.problem, rd(; phi_bc = :zero_boundary); maxiter = 10, stop = nothing, verbose = 0)
    rb = solve!(b.problem, rd(; phi_bc = :zero_boundary, solve = (A, bb, x, ctx, stale) -> (x .= A \ bb)); maxiter = 10,
                stop = nothing, verbose = 0)
    @test get_values(rb.z) ≈ get_values(ra.z) atol = 1e-10
    c = rd_cantilever(; nx = 40, ny = 20, design = :density, frozen = false)
    xs, ys = free_coords(c.Vφ)
    held = findall(i -> xs[i] > 2 - 1e-9 && abs(ys[i] - 0.5) <= 0.1 + 1e-9, eachindex(xs))
    tr = ZTrack(c.z, Vector{Float64}[])
    r = solve!(c.problem, ReactionDiffusion(; τ = 5e-4, KIM2020..., phi_bc = ("load" => 1.0,));   # validation/common/presets.jl
               maxiter = 20, stop = nothing, verbose = 0, monitors = (zt = tr,))
    @test all(isfinite, r.history.objective) && all(zk -> all(==(1.0), zk[held]), tr.zs)
  end

  @testset "15. resuming with phi_bc a checkpoint written without it" begin
    dir = mktempdir()
    a = rd_cantilever(; nx = 30, ny = 15, frozen = false)
    solve!(a.problem, rd(); maxiter = 10, stop = nothing, verbose = 0, checkpoint = Checkpoint(filepath = dir, every = 10))
    b = rd_cantilever(; nx = 30, ny = 15, frozen = false)
    xs, ys = free_coords(b.Vφ)
    bnd = findall(i -> on_box(xs[i], ys[i], (0, 2, 0, 1)), eachindex(xs))
    tr = ZTrack(b.z, Vector{Float64}[])
    rb = solve!(b.problem, rd(; phi_bc = :zero_boundary); maxiter = 14, stop = nothing, verbose = 0, resume = dir,
                monitors = (zt = tr,))
    @test length(tr.zs) == 4 && all(zk -> all(iszero, zk[bnd]), tr.zs) && all(isfinite, rb.history.objective)
  end

  @testset "16. 3D: the boundary of hexahedra and tetrahedra" for simplex in (false, true)
    m3 = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (4, 4, 4))
    simplex && (m3 = simplexify(m3))
    V3 = TestFESpace(m3, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
    p = RDM._phi_bc_nodes(:zero_boundary, V3, -1.0, 1.0)
    @test length(p.idx) == 5^3 - 3^3
  end
end
