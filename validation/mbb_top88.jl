# MBB beam against top88 (Andreassen, Clausen, Schevenels, Lazarov and Sigmund, SMO 43 (2011) 1–16): volume 0.5, p = 3,
# Emin = 1e-9, rmin = 0.04 nelx, OC with move 0.2, stop at max|Δx| ≤ 0.01, on 60×20 and 150×50. Two references:
#
#   Fig. 4 of the paper (published): the compliances of top88 with the PDE (Helmholtz) density filter of section 4.2,
#     237.60 (60×20) and 235.36 (150×50). Same discretisation and formulation as HyperShape's P0 with the cell mean of
#     the filtered field and OC as top88 (the filter of section 4.2 uses r = rmin/(2√3), as HyperShape): the 1 % rule
#     on J.
#   top88.m run in Octave 9.1 with its convolution density filter (ft = 2; data/top88, see its README; the code is not
#     included), whose final compliances are those of Fig. 3 of the paper (233.71, 235.73). Another filter on the same
#     mesh, elements, load and supports: the rule between methods, J_bin within 5 % (J_bin: the design read as 0/1 per
#     cell, E = Emin + (1 − Emin) χ, evaluated with the finite elements of HyperShape's problem for both). The J of
#     top88 in the table is its final design evaluated with those finite elements (the same Q4 elements as top88).
#
# Run: julia --project=validation -t 4 validation/mbb_top88.jl [output directory]
# Writes the fixed report table mbb_top88.csv/.md and the figures mbb_top88_<mesh>.svg (top88 and HyperShape, P0 with
# the cell mean and OC as top88) in the output directory (default validation/results or ENV["HYPERSHAPE_RESULTS"]).
# Exits with code 1 if a check fails.
include(joinpath(@__DIR__, "common", "common.jl"))
const OUTDIR = get(ARGS, 1, results_dir())
const FIG3 = Dict(60 => 233.71, 150 => 235.73)       # Andreassen et al. (2011), Fig. 3: density filter (ft = 2)
const FIG4 = Dict(60 => 237.60, 150 => 235.36)       # Fig. 4: PDE density filter (section 4.2)
println("Julia $(VERSION), threads $(Threads.nthreads()), $(Sys.cpu_info()[1].model)\n")
t88 = (; opt = OC(adapt = nothing), stop = StopCriteria(change_tol = 0.01))
const VARIANTS = (("P0, cell mean, OC as top88", (; t88..., cellmean = true)), ("P0, Gauss points, OC as top88", t88),
                  ("P1, Gauss points, OC as top88", (; t88..., order = 1)), ("P0, cell mean, OC()", (; cellmean = true)),
                  ("P0, Gauss points, SiMPL()", (; opt = SiMPL())), ("P1, Gauss points, SiMPL()", (; opt = SiMPL(), order = 1)))
for (name, kw) in VARIANTS                                                     # compile every variant (the times exclude it)
  c = mbb(; case = :top88, nx = 12, ny = 4, vf = 0.5, rmin = 1.5, Emin = 1e-9, maxiter = 2, kw...)
  c.jbin(c.cells); c.jdens(c.cells)
end
println("| Case | J | vs Fig. 4 (J) | J_bin | vs top88 ft = 2 (J_bin) | V | iterations | status | time |")
println("|---|---|---|---|---|---|---|---|---|")
rows = []
ok = true
mkpath(OUTDIR)
for (nx, ny) in ((60, 20), (150, 50))
  base = (; case = :top88, nx, ny, vf = 0.5, rmin = 0.04nx, Emin = 1e-9)
  ref88 = top88_reference(nx, ny)
  case = "MBB top88 $(nx)×$(ny)"
  # the run of Octave reproduces Fig. 3 (its last printed compliance, to the two decimals of the paper)
  global ok &= round(ref88.c[end], digits = 2) == FIG3[nx]
  Jb88 = NaN
  for (name, kw) in VARIANTS
    c = mbb(; base..., kw...)
    if isnan(Jb88)   # the references: Fig. 4, and the final design of top88.m evaluated with the elements of this problem
      J88, Jb88 = c.jdens(ref88.xPhys), c.jbin(ref88.xPhys)
      V88 = sum(ref88.xPhys) / (nx * ny)
      @printf("| %d×%d Andreassen et al. (2011), Fig. 4, PDE filter | %.2f | — | — | — | — | — | — | — |\n", nx, ny, FIG4[nx])
      @printf("| %d×%d top88.m in Octave (density filter, ft = 2; Fig. 3: %.2f) | %.6g | — | %.6g | — | %.4f | %d | — | — |\n",
              nx, ny, FIG3[nx], J88, Jb88, V88, ref88.iterations)
      push!(rows, reference_row(case, "Andreassen et al. (2011), Fig. 4, PDE density filter (published)"; J = FIG4[nx]))
      push!(rows, reference_row(case, "top88.m in Octave 9.1, density filter ft = 2, rmin = $(0.04nx) (Fig. 3: $(FIG3[nx]))";
                                J = J88, J_bin = Jb88, cells = ref88.xPhys, n = c.n, hcell = c.hcell, V = V88,
                                iterations = ref88.iterations))
      # the figure: the two designs read as 0/1 and the histories of the compliance
      h = c.result.history
      Jh = h.objective[1:h.n]
      it = max(length(ref88.c), length(Jh) + 1)
      design_figure(joinpath(OUTDIR, "mbb_top88_$(nx)x$(ny).svg");
        designs = [(reshape(ref88.xPhys .>= 0.5, nx, ny), ("top88.m (Octave), density filter: $(ref88.iterations) it.",
                                                            @sprintf("J = %.2f, J_bin = %.2f", J88, Jb88))),
                   (reshape(c.cells .>= 0.5, nx, ny), ("HyperShape (Helmholtz, cell mean, OC): $(c.result.iterations) it.",
                                                       @sprintf("J = %.2f, J_bin = %.2f", c.result.objective, c.jbin(c.cells))))],
        curves = [(; x = 1:length(ref88.c), y = ref88.c, color = "black", dash = nothing, label = "top88.m (Octave)"),
                  (; x = 2:length(Jh)+1, y = Jh, color = "#1f6fd1", dash = nothing, label = "HyperShape")],
        vlines = [(length(ref88.c), "black"), (length(Jh) + 1, "#1f6fd1")], ylabel = "compliance",
        xlabel = "iteration (HyperShape's k is top88's k + 1)", xmax = it, xticks = unique([1; 100:100:it-30; it]),
        cell = nx <= 60 ? 6 : 3, stack = true, yfrom = 10, ystep = 10, legend_top = true,
        note = "designs read as 0/1 (χ ≥ ½); vertical dashed lines: the stops")
    end
    # the table: the cell mean with OC as top88 against Fig. 4 (J, 1 %), the others against top88.m (J_bin, 5 %)
    vs88 = c.jbin(c.cells) / Jb88 - 1
    fig4 = startswith(name, "P0, cell mean, OC as top88")
    r = fig4 ? report_row(case, name, c; ref = FIG4[nx]) : report_row(case, name, c; ref = Jb88, basis = :J_bin)
    push!(rows, r)
    @printf("| %d×%d %s | %.6g | %s | %.6g | %+.2f %% | %.4f | %d | %s | %.1f s |\n", nx, ny, name, r.J,
            fig4 ? @sprintf("%+.2f %%", 100 * r.vs_ref) : "—", r.J_bin, 100 * vs88, r.V, r.iterations, r.status, r.time)
    global ok &= c.result.status === :converged && abs(vs88) <= 0.05 && (!fig4 || abs(r.vs_ref) <= 0.01)
  end
  flush(stdout)
end
write_report("mbb_top88", rows; title = "MBB beam against top88 (Andreassen et al. 2011)", dir = OUTDIR)
finish(ok, "top88.m in Octave = Fig. 3; P0 cell mean + OC as top88 within 1 % of Fig. 4; J_bin within 5 % of top88.m (ft = 2); converged")
