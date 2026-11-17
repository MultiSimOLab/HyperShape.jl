# Plain-Julia port of top88 with the PDE filter (Andreassen et al., SMO 43 (2011), section 4.2), ft = 2: an independent reference.
using SparseArrays, LinearAlgebra, Printf
struct T88; nelx; nely; KE; edof; free; F; KF; TF; E0; Emin; penal; end
function T88(nelx, nely, rmin; E0 = 1.0, Emin = 1e-9, nu = 0.3, penal = 3.0)
  A11 = [12 3 -6 -3; 3 12 3 0; -6 3 12 -3; -3 0 -3 12]; A12 = [-6 -3 0 3; -3 -6 -3 -6; 0 -3 -6 3; 3 -6 3 -6]
  B11 = [-4 3 -2 9; 3 -4 -9 4; -2 -9 -4 -3; 9 4 -3 -4]; B12 = [2 -3 4 -9; -3 2 9 -2; 4 9 2 3; -9 -2 3 2]
  KE = 1 / (1 - nu^2) / 24 * ([A11 A12; A12' A11] + nu * [B11 B12; B12' B11])
  nodenrs = reshape(1:(1+nelx)*(1+nely), 1 + nely, 1 + nelx)
  ev = vec(2 * nodenrs[1:end-1, 1:end-1] .+ 1)
  edof = ev .+ [0 1 2nely+2 2nely+3 2nely 2nely+1 -2 -1]
  ndof = 2 * (nelx + 1) * (nely + 1)
  F = zeros(ndof); F[2] = -1.0
  fixed = union(1:2:2*(nely+1), ndof); free = setdiff(1:ndof, fixed)
  Rmin = rmin / 2 / sqrt(3)
  KEF = Rmin^2 * [4 -1 -2 -1; -1 4 -1 -2; -2 -1 4 -1; -1 -2 -1 4] / 6 + [4 2 1 2; 2 4 2 1; 1 2 4 2; 2 1 2 4] / 36
  ef = vec(nodenrs[1:end-1, 1:end-1]) .+ [0 nely+1 nely+2 1]
  ne = nelx * nely; I = Int[]; J = Int[]; S = Float64[]; It = Int[]; Jt = Int[]
  for e in 1:ne, a in 1:4
    push!(It, ef[e, a]); push!(Jt, e)
    for b in 1:4; push!(I, ef[e, a]); push!(J, ef[e, b]); push!(S, KEF[a, b]); end
  end
  KF = cholesky(Symmetric(sparse(I, J, S))); TF = sparse(It, Jt, fill(0.25, length(It)), (nelx + 1) * (nely + 1), ne)
  T88(nelx, nely, KE, edof, free, F, KF, TF, E0, Emin, penal)
end
filt(p::T88, v) = p.TF' * (p.KF \ (p.TF * v))
function fea(p::T88, xPhys)
  ne = p.nelx * p.nely; Ee = p.Emin .+ xPhys .^ p.penal .* (p.E0 - p.Emin)
  I = Vector{Int}(undef, 64ne); J = similar(I); S = Vector{Float64}(undef, 64ne); k = 0
  for e in 1:ne, a in 1:8, b in 1:8; k += 1; I[k] = p.edof[e, a]; J[k] = p.edof[e, b]; S[k] = p.KE[a, b] * Ee[e]; end
  K = sparse(I, J, S); U = zeros(length(p.F)); U[p.free] = cholesky(Symmetric(K[p.free, p.free])) \ p.F[p.free]
  ce = zeros(ne); ue = zeros(8)
  for e in 1:ne; for a in 1:8; ue[a] = U[p.edof[e, a]]; end; ce[e] = dot(ue, p.KE, ue); end
  sum(Ee .* ce), ce
end
compliance(p::T88, xPhys) = fea(p, xPhys)[1]
function top88pde(nelx, nely, volfrac, rmin; maxloop = 3000, kw...)
  p = T88(nelx, nely, rmin; kw...); ne = nelx * nely
  x = fill(volfrac, ne); xPhys = copy(x); change = 1.0; loop = 0; c = 0.0; move = 0.2
  while change > 0.01 && loop < maxloop
    loop += 1
    c, ce = fea(p, xPhys)
    dc = filt(p, -p.penal * (p.E0 - p.Emin) .* xPhys .^ (p.penal - 1) .* ce); dv = filt(p, ones(ne))
    l1 = 0.0; l2 = 1e9; xnew = x
    while (l2 - l1) / (l1 + l2) > 1e-3
      lmid = 0.5 * (l2 + l1)
      xnew = max.(0, max.(x .- move, min.(1, min.(x .+ move, x .* sqrt.(-dc ./ dv ./ lmid)))))
      xPhys = filt(p, xnew)
      sum(xPhys) > volfrac * ne ? (l1 = lmid) : (l2 = lmid)
    end
    change = maximum(abs, xnew - x); x = xnew
  end
  (; c, loop, xPhys, x, p)
end
# cell means in Gridap ordering (x fastest, y upwards) -> top88 ordering (columns, y downwards)
gridap_to_top88(ρc, nx, ny) = vec(reverse(reshape(ρc, nx, ny)', dims = 1))
if abspath(PROGRAM_FILE) == @__FILE__
  println("Julia $(VERSION)")
  for (nx, ny, r) in ((60, 20, 2.4), (150, 50, 6.0), (300, 100, 12.0), (300, 100, 16.0))[1:parse(Int, get(ARGS, 1, "4"))]
    t = @elapsed res = top88pde(nx, ny, 0.5, r)
    @printf("top88pde %dx%d rmin %.1f: c = %.4f (last evaluated), c(final design) = %.4f, loops %d, %.1f s\n", nx, ny, r, res.c, compliance(res.p, res.xPhys), res.loop, t); flush(stdout)
  end
end
