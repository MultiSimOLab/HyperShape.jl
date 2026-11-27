# MBB beam against top88 (Andreassen et al., SMO 43 (2011) 1–16): volume 0.5, p = 3, Emin = 1e-9, rmin = 0.04 nelx,
# OC with move 0.2, stop at max|Δx| ≤ 0.01, on 60×20 and 150×50. The reference is the code top88.m run in Octave 9.1
# (data/top88, see its README; the code is not included) with its density filter (ft = 2: a convolution filter of
# radius rmin in element units). HyperShape uses a Helmholtz filter with the usual equivalence r = rmin/(2√3)
# (Lazarov and Sigmund 2011; `HelmholtzFilter(dΩ; rmin)`), so this is a comparison between two filters on the same
# mesh, elements, load and supports: the rule between methods applies, J_bin within 5 % (J_bin: the design read as
# 0/1 per cell, E = Emin + (1 − Emin) χ, evaluated with the finite elements of HyperShape's problem for both). The J of
# top88 in the table is its final design evaluated with those finite elements (the same Q4 elements as top88: equal
# to its own evaluation to round-off).
# Run: julia --project=validation -t 4 validation/mbb_top88.jl [output directory]
# Writes the fixed report table mbb_top88.csv/.md and the figures mbb_top88_<mesh>.svg (top88 and HyperShape, P0 with
# the cell mean and OC as top88) in the output directory (default validation/results or ENV["HYPERSHAPE_RESULTS"]).
include(joinpath(@__DIR__, "common", "common.jl"))
const OUTDIR = get(ARGS, 1, results_dir())
println("Julia $(VERSION), threads $(Threads.nthreads()), $(Sys.cpu_info()[1].model)\n")
mbb(; case = :top88, nx = 12, ny = 4, vf = 0.5, rmin = 1.5, Emin = 1e-9, maxiter = 2)        # compile
for kw in ((; cellmean = true), (; order = 1), (; opt = SiMPL()), (; opt = SiMPL(), order = 1))   # compile the other variants
  mbb(; case = :top88, nx = 12, ny = 4, vf = 0.5, rmin = 1.5, Emin = 1e-9, maxiter = 2, kw...)
end
t88 = (; opt = OC(adapt = nothing), stop = StopCriteria(change_tol = 0.01))
println("| Case | J | J_bin | J_bin vs top88 | V | iterations | status | time |\n|---|---|---|---|---|---|---|---|")
rows = []
ok = true
mkpath(OUTDIR)
for (nx, ny) in ((60, 20), (150, 50))
  base = (; case = :top88, nx, ny, vf = 0.5, rmin = 0.04nx, Emin = 1e-9)
  ref88 = top88_reference(nx, ny)
  case = "MBB top88 $(nx)×$(ny)"
  Jb88 = NaN
  for (name, kw) in (("P0, cell mean, OC as top88", (; t88..., cellmean = true)), ("P0, Gauss points, OC as top88", t88),
                     ("P1, Gauss points, OC as top88", (; t88..., order = 1)), ("P0, cell mean, OC()", (; cellmean = true)),
                     ("P0, Gauss points, SiMPL()", (; opt = SiMPL())), ("P1, Gauss points, SiMPL()", (; opt = SiMPL(), order = 1)))
    c = mbb(; base..., kw...)
    if isnan(Jb88)   # the reference: the final design of top88, evaluated with the finite elements of this problem
      J88, Jb88 = c.jdens(ref88.xPhys), c.jbin(ref88.xPhys)
      V88 = sum(ref88.xPhys) / (nx * ny)
      @printf("| %d×%d top88.m in Octave (density filter, ft = 2) | %.6g | %.6g | — | %.4f | %d | — | — |\n", nx, ny, J88, Jb88,
              V88, ref88.iterations)
      push!(rows, reference_row(case, "top88.m in Octave 9.1, density filter ft = 2, rmin = $(0.04nx) (reference)";
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
        cell = nx <= 60 ? 6 : 3, stack = true, yfrom = 10, ystep = 10, legend_top = true, note = "designs read as 0/1 (χ ≥ ½); vertical dashed lines: the stops")
    end
    r = report_row(case, name, c; ref = Jb88, basis = :J_bin)
    push!(rows, r)
    @printf("| %d×%d %s | %.6g | %.6g | %+.2f %% | %.4f | %d | %s | %.1f s |\n", nx, ny, name, r.J, r.J_bin, 100 * r.vs_ref,
            r.V, r.iterations, r.status, r.time)
    global ok &= c.result.status === :converged && abs(r.vs_ref) <= 0.05
  end
  flush(stdout)
end
println(ok ? "\nall checks passed (|J_bin − J_bin of top88| ≤ 5 %, converged)" : "\nSOME CHECK FAILED")
write_report("mbb_top88", rows; title = "MBB beam against top88 (Octave, density filter ft = 2)", dir = OUTDIR)
