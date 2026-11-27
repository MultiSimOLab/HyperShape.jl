# The cantilever of levelset88 (Otomori, Yamada, Izui and Nishiwaki 2015, Struct. Multidisc. Optim. 51:1159) with
# ReactionDiffusion as it is (the public API, no code written for this case), against the literal code run in Octave
# (data/levelset88, see its README). Not a reproduction of levelset88: the method of HyperShape is the closest existing
# choice, with the parameters that the paper fixes and the defaults of the API for the rest:
#
#   fixed by the paper: 80 × 64 unit elements, clamped left side, 5 unit nodal forces in the middle of the right side,
#     E = 1, Emin = 1e-4, ν = 0.3 (plane stress), Vmax = 0.5, τ = 2e-4 (τ ℓ² = τ N_e, as their T = NN/Δt + τ N_e NNdif),
#     Δt = 0.1, consistent mass, φ = 0 on the boundary (phi_bc = :zero_boundary), void weight 1e-4 of the sensitivity,
#     a volume going down over 100 iterations (max_step = 0.5/100 = 0.005 of |D|), at most 200 iterations;
#   HyperShape instead of levelset88: Ersatz() (cut cells) instead of a binary element, the exact derivative of the
#     compliance weighted by the material (:physical) instead of the topological derivative of the spherical hole, the
#     exact multiplier of the volume (ExactVolume) instead of their exponential rule, L1Mean() and the default stop of
#     ReactionDiffusion instead of their criterion.
#
# history.objective[k] is the compliance after iteration k, which levelset88 records at iteration k + 1: the table
# and the figure compare HyperShape's k with Octave's k + 1. The volumes are the fraction of Ersatz() (the one the
# constraint sees) and, as levelset88 measures it, the fraction of elements whose mean nodal φ is positive. Besides the
# compliance that HyperShape optimises (cut cells of Ersatz()), the compliance of its design read as levelset88 reads
# it (binary elements, E = Emin + (1 − Emin) χ_e, the same mesh and load: plain Gridap) is given at every iteration:
# that one compares like for like with Octave. The same evaluation of Octave's design at its stop is printed as a
# check of the model (it must give Octave's J).
#
# julia --project=validation -t 4 validation/otomori_levelset88.jl [output directory]     # about 1 min
# writes otomori_levelset88.svg (both designs and both histories of the compliance) in the output directory (default:
# the current directory)
using Gridap, HyperFEM, HyperShape, Printf

const DATA = joinpath(@__DIR__, "data", "levelset88")
const OUTDIR = get(ARGS, 1, pwd())
const NX, NY, VMAX, TAU = 80, 64, 0.5, 2e-4

# the Octave reference: J = F'U and the binary volume of the design before the update of each iteration, the stop of
# Otomori's criterion and the binary design there (64 × 80, column-major, row 1 at the top)
function octave_reference()
  lines = readlines(joinpath(DATA, "ref_ls88_80x64_tau2e-4_hist.csv"))
  stop = parse(Int, split(lines[1], "stop_it=")[2])
  rows = [parse.(Float64, split(l, ",")) for l in lines[2:end]]
  J, V = [r[2] for r in rows], [r[3] for r in rows]
  s = parse.(Float64, split(strip(read(joinpath(DATA, "ref_ls88_80x64_tau2e-4_str_stop.csv"), String)), ","))
  S = reshape(s, NY, NX)                                    # S[r, c]: row r from the top
  str = [S[NY-j+1, i] > 0.5 for i in 1:NX, j in 1:NY]            # str[i, j]: element i along x, j along y from the bottom
  (; J, V, stop, str)
end

# the binary design of levelset88 from the nodal values: element material when the mean of its 4 nodes is positive
struct BinaryDesign
  z::OptVariables
  cells::Vector{NTuple{4,Int}}
  snap::Int                                                  # the iteration whose design is kept
  kept::Vector{Vector{Bool}}
  k::Base.RefValue{Int}
end
binary(b::BinaryDesign, v = get_values(b.z)) = [sum(v[i] for i in c) / 4 > 0 for c in b.cells]
function HyperShape.Methods.monitor_value(b::BinaryDesign)
  b.k[] += 1
  str = binary(b)
  b.k[] == b.snap && push!(b.kept, str)
  count(str) / length(str)
end
# the compliance of a binary design with the model of levelset88 (elements of E = Emin + (1 − Emin) χ_e)
struct BinaryCompliance{B,F}
  bd::B
  J::F
end
HyperShape.Methods.monitor_value(c::BinaryCompliance) = c.J(binary(c.bd))

struct ErsatzVolume{T,M}
  θ::T
  dΩ::M
end
HyperShape.Methods.monitor_value(e::ErsatzVolume) = sum(∫(e.θ.χ)e.dΩ) / (NX * NY)

function run_levelset88(ref)
  model = CartesianDiscreteModel((0, NX, 0, NY), (NX, NY))
  labels = get_face_labeling(model)
  r = round(Int, NY / 32)
  add_tag_from_vertex_filter!(labels, "clamp", model, x -> x[1] < 1e-9)
  add_tag_from_vertex_filter!(labels, "load", model, x -> x[1] > NX - 1e-9 && abs(x[2] - NY / 2) <= r + 1e-9)
  Ω = Triangulation(model)
  δp = DiracDelta{0}(model; tags = "load")                  # unit nodal forces, as F of levelset88
  bc = BoundaryConditions(BoundaryCondition("clamp", VectorValue(0.0, 0.0)))
  V = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1), bc; conformity = :H1)
  U = TrialFESpace(V, bc, 1.0)
  Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
  z = OptVariables(φ = FieldVariable(Vφ; bounds = (-1.0, 1.0), init = 1.0))
  θ = PhysicalFields(DesignFields(z; φ = Identity()); χ = :φ => Ersatz())
  dΩ = design_measures(θ; degree = 4)
  ν, Emin = 0.3, 1e-4
  λ, μ = ν / (1 - ν^2), 1 / (2(1 + ν))
  σ(e) = λ * tr(e) * one(e) + 2μ * e
  E(χ) = Emin + χ * (1 - Emin)
  t = VectorValue(0.0, 1.0)
  a(u, v) = ∫((E ∘ θ.χ) * (ε(v) ⊙ (σ ∘ ε(u))))dΩ.domain
  S = StateOperator(StaticLinearModel(v -> δp(t ⋅ v), a, U, V, bc; ls = LUSolver(), backend = :compiled), θ;
        adjoint = AdjointProblem(sensitivity = (u, q) -> (δ -> ∫((1 - Emin) * (ε(q) ⊙ (σ ∘ ε(u))) * δ)dΩ.domain)))
  C = Functional((u, θ) -> δp(t ⋅ u), S; dJdu = SelfAdjoint(), scale = 1.0)      # J = F'U
  problem = Optimization(z; objective = C, constraints = (volume = Volume(θ, dΩ; max_fraction = VMAX),))
  method = ReactionDiffusion(; τ = TAU, sensitivity = :physical, Δt = 0.1, mass = :consistent, void_weight = 1e-4,
                             phi_bc = :zero_boundary, volume = ExactVolume(max_step = (1 - VMAX) / 100))
  # the model of levelset88 for a binary design: plain Gridap, the same mesh, supports and load
  Vb = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1); conformity = :H1, dirichlet_tags = "clamp")
  Ub = TrialFESpace(Vb, VectorValue(0.0, 0.0))
  V0 = FESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
  dΩb = Measure(Ω, 2)
  Jbin = str -> begin
    Ee = FEFunction(V0, [Emin + (s ? 1 - Emin : 0.0) for s in vec(str)])
    uh = solve(AffineFEOperator((u, v) -> ∫(Ee * (ε(v) ⊙ (σ ∘ ε(u))))dΩb, v -> δp(t ⋅ v), Ub, Vb))
    sum(δp(t ⋅ uh))
  end
  ids = get_cell_dof_ids(Vφ)
  cells = [ntuple(a -> Int(ids[c][a]), 4) for c in 1:num_cells(Ω)]          # no frozen values: all of them are values of z
  bd = BinaryDesign(z, cells, ref.stop - 1, Vector{Bool}[], Ref(0))
  tt = @elapsed res = solve!(problem, method; maxiter = 200, verbose = 0, monitors = (vbin = bd, vers = ErsatzVolume(θ, dΩ.domain),
                                                                                       jbin = BinaryCompliance(bd, Jbin)))
  str = reshape(binary(bd), NX, NY)                         # Gridap's Cartesian cells: x first
  kept = isempty(bd.kept) ? nothing : reshape(bd.kept[1], NX, NY)
  (; res, str, kept, time = tt, Jbin)
end

# interior holes (by faces, not touching the boundary) of a binary design
function holes(str)
  void, seen, n = .!str, falses(size(str)), 0
  for j in axes(str, 2), i in axes(str, 1)
    (void[i, j] && !seen[i, j]) || continue
    stack, inner = [(i, j)], true
    seen[i, j] = true
    while !isempty(stack)
      (a, b) = pop!(stack)
      for (p, q) in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1))
        checkbounds(Bool, str, p, q) || (inner = false; continue)
        (void[p, q] && !seen[p, q]) || continue
        seen[p, q] = true
        push!(stack, (p, q))
      end
    end
    inner && (n += 1)
  end
  n
end

# The figure (plain SVG, no plotting package): the two designs and the two histories of the compliance
function figure(path, ref, hs)
  cell, W0 = 4, 20
  pw, ph = NX * cell, NY * cell
  io = IOBuffer()
  W, H = 3W0 + 2pw + 560, ph + 90
  println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="sans-serif" font-size="13" shape-rendering="crispEdges">""")
  println(io, """<rect width="$W" height="$H" fill="white"/>""")
  function design(x0, str, title)
    println(io, """<text x="$x0" y="15">$(title[1])</text><text x="$x0" y="31">$(title[2])</text>""")
    println(io, """<rect x="$x0" y="40" width="$pw" height="$ph" fill="none" stroke="#999"/>""")
    for j in 1:NY
      y = 40 + (NY - j) * cell
      i = 1
      while i <= NX                                       # runs of material along a row
        str[i, j] || (i += 1; continue)
        i0 = i
        while i <= NX && str[i, j]
          i += 1
        end
        # a slight overlap with the next row: no gaps between rows
        println(io, """<rect x="$(x0 + (i0 - 1) * cell)" y="$y" width="$((i - i0) * cell)" height="$(cell + 0.5)" fill="black"/>""")
      end
    end
  end
  r = hs.res
  Jb = r.history.monitors.jbin
  design(W0, ref.str, ("levelset88 (Octave): stop at it. $(ref.stop)", "J = $(@sprintf("%.2f", ref.J[ref.stop]))"))
  design(2W0 + pw, hs.str, ("HyperShape: $(r.iterations) it. (Ersatz J = $(@sprintf("%.2f", r.objective)))",
                            "as binary elements: J = $(@sprintf("%.2f", Jb[end]))"))
  # the histories, against the iteration of levelset88 (HyperShape's k is Octave's k + 1)
  x0, y0, gw, gh = 3W0 + 2pw + 100, 40, 420, ph - 30
  Jh = r.history.objective
  Jmin = min(minimum(ref.J), minimum(Jh), minimum(Jb))
  Jmax = max(maximum(ref.J), maximum(Jh), maximum(Jb))
  lo, hi = 50 * floor(Jmin / 50), 50 * ceil(Jmax / 50)
  X(it) = x0 + gw * (it - 1) / 199
  Y(J) = y0 + gh * (hi - J) / (hi - lo)
  println(io, """<text x="$x0" y="22">compliance F'U</text>""")
  println(io, """<rect x="$x0" y="$y0" width="$gw" height="$gh" fill="none" stroke="#999"/>""")
  for J in lo:50:hi
    println(io, """<line x1="$(x0 - 4)" x2="$x0" y1="$(Y(J))" y2="$(Y(J))" stroke="#999"/><text x="$(x0 - 8)" y="$(Y(J) + 4)" text-anchor="end">$(Int(J))</text>""")
  end
  for it in (1, 50, 100, 150, 200)
    println(io, """<line x1="$(X(it))" x2="$(X(it))" y1="$(y0 + gh)" y2="$(y0 + gh + 4)" stroke="#999"/><text x="$(X(it))" y="$(y0 + gh + 18)" text-anchor="middle">$it</text>""")
  end
  println(io, """<text x="$(x0 + gw / 2)" y="$(y0 + gh + 36)" text-anchor="middle">iteration of levelset88</text>""")
  pts(xs, ys) = join((@sprintf("%.1f,%.1f", X(a), Y(b)) for (a, b) in zip(xs, ys)), " ")
  println(io, """<polyline fill="none" stroke="black" stroke-width="1.5" points="$(pts(1:length(ref.J), ref.J))"/>""")
  println(io, """<polyline fill="none" stroke="#1f6fd1" stroke-width="1.5" points="$(pts(2:length(Jh)+1, Jh))"/>""")
  println(io, """<polyline fill="none" stroke="#1f6fd1" stroke-width="1.5" stroke-dasharray="5,3" points="$(pts(2:length(Jb)+1, Jb))"/>""")
  println(io, """<line x1="$(X(ref.stop))" x2="$(X(ref.stop))" y1="$y0" y2="$(y0 + gh)" stroke="black" stroke-dasharray="3,3"/>""")
  println(io, """<line x1="$(X(r.iterations + 1))" x2="$(X(r.iterations + 1))" y1="$y0" y2="$(y0 + gh)" stroke="#1f6fd1" stroke-dasharray="3,3"/>""")
  println(io, """<line x1="$(x0 + 190)" x2="$(x0 + 215)" y1="$(y0 + gh - 60)" y2="$(y0 + gh - 60)" stroke="black" stroke-width="1.5"/><text x="$(x0 + 220)" y="$(y0 + gh - 56)">levelset88 (Octave)</text>""")
  println(io, """<line x1="$(x0 + 190)" x2="$(x0 + 215)" y1="$(y0 + gh - 40)" y2="$(y0 + gh - 40)" stroke="#1f6fd1" stroke-width="1.5"/><text x="$(x0 + 220)" y="$(y0 + gh - 36)">HyperShape (Ersatz)</text>""")
  println(io, """<line x1="$(x0 + 190)" x2="$(x0 + 215)" y1="$(y0 + gh - 20)" y2="$(y0 + gh - 20)" stroke="#1f6fd1" stroke-width="1.5" stroke-dasharray="5,3"/><text x="$(x0 + 220)" y="$(y0 + gh - 16)">HyperShape as binary elements</text>""")
  println(io, """<text x="$x0" y="$(y0 + gh + 52)" font-size="11">vertical dashed lines: the stops (Otomori's criterion; the default stop of ReactionDiffusion)</text>""")
  println(io, "</svg>")
  write(path, take!(io))
end

ref = octave_reference()
println("Cantilever of levelset88, 80 × 64, Vmax = 0.5, τ = 2e-4 · ", Sys.cpu_info()[1].model, ", ", Threads.nthreads(),
        " threads, Julia ", VERSION, "\n")
hs = run_levelset88(ref)
r = hs.res
h = r.history
Jh, vbin, vers = h.objective, h.monitors.vbin, h.monitors.vers
Jb = h.monitors.jbin
println("| iteration (levelset88) | J HyperShape (Ersatz) | J HyperShape (binary elements) | J Octave | difference (binary) | volume HyperShape (Ersatz) | volume HyperShape (elements) | volume Octave (elements) |")
println("|---|---|---|---|---|---|---|---|")
ks = sort!(unique!(filter(k -> k <= length(Jh), [1; 10:10:200; ref.stop - 1; length(Jh)])))
for k in ks
  it = k + 1
  it <= length(ref.J) || continue
  @printf("| %d%s | %.4f | %.4f | %.4f | %+.2f %% | %.4f | %.4f | %.4f |\n", it, it == ref.stop ? " (stop of Octave)" : k == length(Jh) ? " (end of HyperShape)" : "",
          Jh[k], Jb[k], ref.J[it], 100 * (Jb[k] / ref.J[it] - 1), vers[k], vbin[k], ref.V[it])
end
Js = ref.J[ref.stop]
println()
@printf("levelset88 (Octave): stop at iteration %d, J = %.4f, volume %.4f, %d interior holes\n", ref.stop, Js, ref.V[ref.stop], holes(ref.str))
Jcheck = hs.Jbin(ref.str)
@printf("check of the binary model: Octave's design at its stop evaluated here, J = %.6f (Octave %.6f, relative difference %.1e)\n",
        Jcheck, Js, abs(Jcheck / Js - 1))
@printf("HyperShape: %s (%s) after %d iterations (= iteration %d of levelset88), J = %.4f with Ersatz() (%+.2f %% against the stop of Octave), J = %.4f as binary elements (%+.2f %%), volume %.4f (Ersatz) / %.4f (elements), %d interior holes, %.1f s\n",
        r.status, r.reason, r.iterations, r.iterations + 1, r.objective, 100 * (r.objective / Js - 1), Jb[end],
        100 * (Jb[end] / Js - 1), vers[end], vbin[end], holes(hs.str), hs.time)
@printf("elements of a different phase at the end of each run: %d of %d (%.1f %%)\n", count(hs.str .!= ref.str), NX * NY,
        100 * count(hs.str .!= ref.str) / (NX * NY))
hs.kept === nothing ||
  @printf("at the same iteration (%d of levelset88): %d elements of a different phase (%.1f %%), J HyperShape %.4f (Ersatz) / %.4f (binary elements)\n", ref.stop,
          count(hs.kept .!= ref.str), 100 * count(hs.kept .!= ref.str) / (NX * NY), Jh[ref.stop-1], Jb[ref.stop-1])
mkpath(OUTDIR)
fig = joinpath(OUTDIR, "otomori_levelset88.svg")
figure(fig, ref, hs)
println("figure: ", fig)
