# Starting designs with holes (hole_seed): a pattern of holes in the free values of a design field, for methods that
# only move the boundary of the material and for studies of the dependence of a design on its start.

# Points in a uniform grid of boxes of side ≥ the search radius (linked lists in two integer vectors): `_nearest2` is
# exact for distances up to the side of the boxes, and larger otherwise. Memory linear in the points.
struct _Buckets{D}
  side::Float64
  lo::NTuple{D,Float64}
  n::NTuple{D,Int}
  head::Vector{Int}
  next::Vector{Int}
  pts::Vector{NTuple{D,Float64}}
end

function _Buckets(lo::NTuple{D,Float64}, hi::NTuple{D,Float64}, radius::Float64, npoints::Int) where {D}
  vol = prod(max.(hi .- lo, radius))
  side = max(radius, (vol / max(npoints, 1))^(1 / D))          # never more boxes than points
  n = ntuple(k -> max(1, ceil(Int, (hi[k] - lo[k]) / side)), D)
  _Buckets{D}(side, lo, n, zeros(Int, prod(n)), Int[], NTuple{D,Float64}[])
end

@inline _box(b::_Buckets{D}, x) where {D} = ntuple(k -> clamp(floor(Int, (x[k] - b.lo[k]) / b.side) + 1, 1, b.n[k]), D)

@inline function _lin(b::_Buckets{D}, I) where {D}
  l, s = 0, 1
  for k in 1:D
    l += (I[k] - 1) * s
    s *= b.n[k]
  end
  l + 1
end

function _push!(b::_Buckets, x)
  push!(b.pts, x)
  l = _lin(b, _box(b, x))
  push!(b.next, b.head[l])
  b.head[l] = length(b.pts)
  b
end

# the smallest squared distance from x to the points of the 3^D boxes around it (Inf if there is none)
@inline function _nearest2(b::_Buckets{D}, x) where {D}
  I = _box(b, x)
  best = Inf
  for J in CartesianIndices(ntuple(k -> max(1, I[k] - 1):min(b.n[k], I[k] + 1), D))
    p = b.head[_lin(b, Tuple(J))]
    while p != 0
      q = b.pts[p]
      d2 = 0.0
      for k in 1:D
        d2 += (x[k] - q[k])^2
      end
      best = min(best, d2)
      p = b.next[p]
    end
  end
  best
end

# the index (in insertion order) of the nearest point to x, searching rings of boxes until no box outside can be
# nearer (exact); 0 if there are no points
function _nearest_index(b::_Buckets{D}, x) where {D}
  isempty(b.pts) && return 0
  I = _box(b, x)
  best, ibest = Inf, 0
  for ring in 0:maximum(b.n)
    for J in CartesianIndices(ntuple(k -> max(1, I[k] - ring):min(b.n[k], I[k] + ring), D))
      maximum(k -> abs(J[k] - I[k]), 1:D) == ring || continue            # the shell of this ring only
      p = b.head[_lin(b, Tuple(J))]
      while p != 0
        q = b.pts[p]
        d2 = 0.0
        for k in 1:D
          d2 += (x[k] - q[k])^2
        end
        d2 < best && ((best, ibest) = (d2, p))
        p = b.next[p]
      end
    end
    ibest != 0 && best <= (ring * b.side)^2 && break
  end
  ibest
end

_tup(p, ::Val{D}) where {D} = ntuple(k -> Float64(p[k]), D)

_seed_tags(t) = t === nothing ? () : t isa Union{AbstractString,Symbol} ? (String(t),) : Tuple(String(s) for s in t)

# ---------------------------------------------------------------------------------------------------------------
# Symmetries: isometries x ↦ A x + b (A orthogonal), the finite group they generate

struct _Isometry{D}
  A::NTuple{D,NTuple{D,Float64}}      # rows
  b::NTuple{D,Float64}
end

@inline _apply(g::_Isometry{D}, x) where {D} = ntuple(k -> g.b[k] + sum(g.A[k][j] * x[j] for j in 1:D), D)
_compose(g::_Isometry{D}, h::_Isometry{D}) where {D} =        # g ∘ h
  _Isometry{D}(ntuple(i -> ntuple(j -> sum(g.A[i][k] * h.A[k][j] for k in 1:D), D), D), _apply(g, h.b))
function _inverse(g::_Isometry{D}) where {D}
  At = ntuple(i -> ntuple(j -> g.A[j][i], D), D)
  _Isometry{D}(At, ntuple(i -> -sum(At[i][k] * g.b[k] for k in 1:D), D))
end
_identity(::Val{D}) where {D} = _Isometry{D}(ntuple(i -> ntuple(j -> Float64(i == j), D), D), ntuple(_ -> 0.0, D))
_close(g::_Isometry, h::_Isometry, tol) = all(abs.(g.b .- h.b) .<= tol) && all(i -> all(abs.(g.A[i] .- h.A[i]) .<= 1e-9), eachindex(g.A))

function _unit(v, what, ::Val{D}) where {D}
  length(v) == D || throw(ArgumentError("hole_seed: the $what of a symmetry must have $D coordinates"))
  t = _tup(v, Val(D))
  n = sqrt(sum(abs2, t))
  (isfinite(n) && n > 0) || throw(ArgumentError("hole_seed: the $what of a symmetry must be a nonzero vector"))
  t ./ n
end

function _generator(s, ::Val{D}) where {D}
  s isa NamedTuple ||
    throw(ArgumentError("hole_seed: each symmetry is a named tuple: (point = p, normal = n) for a reflection, " *
                        "(point = p, order = n) for a rotation in 2D, (point = p, axis = a, order = n) in 3D"))
  haskey(s, :point) || throw(ArgumentError("hole_seed: a symmetry needs a point (point = …)"))
  length(s.point) == D || throw(ArgumentError("hole_seed: the point of a symmetry must have $D coordinates"))
  p = _tup(s.point, Val(D))
  if haskey(s, :normal)
    (haskey(s, :order) || haskey(s, :axis)) && throw(ArgumentError("hole_seed: a symmetry is a reflection (normal) or a rotation (order), not both"))
    n = _unit(s.normal, "normal", Val(D))
    A = ntuple(i -> ntuple(j -> Float64(i == j) - 2n[i] * n[j], D), D)
  elseif haskey(s, :order)
    (s.order isa Integer && s.order >= 2) || throw(ArgumentError("hole_seed: the order of a rotation must be an integer ≥ 2, got $(s.order)"))
    θ = 2π / s.order
    c, si = cos(θ), sin(θ)
    if D == 2
      haskey(s, :axis) && throw(ArgumentError("hole_seed: a rotation in 2D has no axis (it turns about the point)"))
      A = ((c, -si), (si, c))
    elseif D == 3
      haskey(s, :axis) || throw(ArgumentError("hole_seed: a rotation in 3D needs an axis (axis = …)"))
      a = _unit(s.axis, "axis", Val(3))
      K = ((0.0, -a[3], a[2]), (a[3], 0.0, -a[1]), (-a[2], a[1], 0.0))
      A = ntuple(i -> ntuple(j -> c * (i == j) + si * K[i][j] + (1 - c) * a[i] * a[j], 3), 3)
    else
      throw(ArgumentError("hole_seed: rotations in 2D and 3D only"))
    end
  else
    throw(ArgumentError("hole_seed: a symmetry needs normal = … (a reflection) or order = … (a rotation)"))
  end
  A = A::NTuple{D,NTuple{D,Float64}}
  _Isometry{D}(A, ntuple(i -> p[i] - sum(A[i][j] * p[j] for j in 1:D), D))
end

# the group generated by the symmetries (identity first), its inverses, the generators and a generic direction w:
# the fundamental region is the set of points x with w⋅x ≥ w⋅g(x) for every g (one representative per orbit)
function _symmetry_group(symmetry, ::Val{D}, L) where {D}
  gens = symmetry === nothing ? () : symmetry isa NamedTuple ? (symmetry,) : Tuple(symmetry)
  gs = [_generator(s, Val(D)) for s in gens]
  tol = 1e-9 * L
  els = [_identity(Val(D))]
  i = 1
  while i <= length(els)
    for g in gs
      h = _compose(g, els[i])
      any(e -> _close(e, h, tol), els) && continue
      push!(els, h)
      length(els) > 192 &&
        throw(ArgumentError("hole_seed: the symmetries do not generate a finite group (planes at angles that are not π/m, " *
                            "rotations about different axes)"))
    end
    i += 1
  end
  w = D == 2 ? (0.8320502943378437, 0.5547001962252291) : (0.8017837257372732, 0.5345224838248488, 0.2672612419124244)
  (; G = els, Ginv = map(_inverse, els), gens = gs, w = ntuple(k -> w[k], D))
end

@inline _dotw(S, x) = sum(S.w .* x)
_in_fundamental(S, x, tol) = (wx = _dotw(S, x); all(g -> _dotw(S, _apply(g, x)) <= wx + tol, S.G))

# the distinct images of c (points closer than tol are one)
function _orbit(S, c, tol)
  o = [c]
  for g in S.G
    p = _apply(g, c)
    any(q -> sum(abs2, p .- q) <= tol^2, o) || push!(o, p)
  end
  o
end

# the centre moved to the mean of its images closer than d: onto the plane or the axis it is near to
function _snap(S, c, d)
  length(S.G) == 1 && return c
  s, n = c .* 0.0, 0
  for g in S.G
    p = _apply(g, c)
    sum(abs2, p .- c) < d^2 || continue
    s = s .+ p
    n += 1
  end
  s ./ n
end

# ---------------------------------------------------------------------------------------------------------------
# What the seed reads from the space: cell coordinates, centroids, sizes and measures, the frozen cells (all their
# values are Dirichlet values, as for FieldVariable), the box of the mesh
function _seed_geometry(V)
  Ω = get_triangulation(V)
  model = get_background_model(Ω)
  D = num_cell_dims(Ω)
  X = get_cell_coordinates(Ω)
  nc = length(X)
  cen = Vector{NTuple{D,Float64}}(undef, nc)
  h = Vector{Float64}(undef, nc)
  lo, hi = ntuple(_ -> Inf, D), ntuple(_ -> -Inf, D)
  for (c, xs) in enumerate(X)
    cen[c] = _tup(sum(xs) / length(xs), Val(D))
    m = 0.0
    for a in 1:length(xs)
      p = _tup(xs[a], Val(D))
      lo, hi = min.(lo, p), max.(hi, p)
      for b in a+1:length(xs)
        m = max(m, norm(xs[a] - xs[b]))
      end
    end
    # the longest distance between two vertices; the diagonal of an n-cube divided by √D
    h[c] = length(xs) == 2^D && D > 1 ? m / sqrt(D) : m
  end
  meas = collect(Float64, get_cell_measure(Ω))
  ids = get_cell_dof_ids(V)
  frozen = BitVector([all(<(0), cell) for cell in ids])
  (; Ω, model, D, X, cen, h, meas, frozen, ids, lo, hi, hmean = sum(h) / nc, hmax = maximum(h))
end

# the vertices of the entities of the tags (of any dimension, cells included)
function _tag_vertices(model, tags, ::Val{D}) where {D}
  labels = get_face_labeling(model)
  topo = Gridap.Geometry.get_grid_topology(model)
  xv = Gridap.Geometry.get_vertex_coordinates(topo)
  pts = NTuple{D,Float64}[]
  isempty(tags) && return pts
  mask = falses(length(xv))
  for t in tags, d in 0:D
    m = Gridap.Geometry.get_face_mask(labels, t, d)
    any(m) || continue
    f2v = Gridap.Geometry.get_faces(topo, d, 0)
    for f in findall(m), v in f2v[f]
      mask[v] = true
    end
  end
  for v in findall(mask)
    push!(pts, _tup(xv[v], Val(D)))
  end
  pts
end

function _check_tags(model, tags, kw)
  names = get_face_labeling(model).tag_to_name
  for t in tags
    t in names || throw(ArgumentError("hole_seed: no tag named $t (in $kw) in the labels of the model"))
  end
end

# the points of a lattice (:lattice square/cubic of step s, symmetric in the box; :hex close-packed with neighbours s
# apart, a point at the centre of the box; `origin` a point of the lattice), extended `ext` beyond the box
function _lattice_points(pattern, lo, hi, s, origin, ext, ::Val{D}) where {D}
  o = origin !== nothing ? _tup(origin, Val(D)) :
      pattern === :hex ? ntuple(k -> (lo[k] + hi[k]) / 2, D) :
      ntuple(k -> (lo[k] + hi[k]) / 2 - (max(1, round(Int, (hi[k] - lo[k]) / s)) - 1) / 2 * s, D)
  cell, basis = if pattern === :lattice || D ∉ (2, 3)
    ntuple(_ -> s, D), (ntuple(_ -> 0.0, D),)
  elseif D == 2
    (s, sqrt(3) * s), ((0.0, 0.0), (s / 2, sqrt(3) * s / 2))
  else
    a = sqrt(2) * s
    (a, a, a), ((0.0, 0.0, 0.0), (a / 2, a / 2, 0.0), (a / 2, 0.0, a / 2), (0.0, a / 2, a / 2))
  end
  ranges = ntuple(k -> (floor(Int, (lo[k] - ext - o[k]) / cell[k]) - 1):(ceil(Int, (hi[k] + ext - o[k]) / cell[k]) + 1), D)
  pts = NTuple{D,Float64}[]
  for I in CartesianIndices(ranges), b in basis
    push!(pts, ntuple(k -> o[k] + I[k] * cell[k] + b[k], D))
  end
  pts
end

# the distinct images of the centres (closer than tol are one), the centre each comes from and its element of the group
function _images(cs::Vector{NTuple{D,Float64}}, S, tol) where {D}
  gen = NTuple{D,Float64}[]; owner = Int[]; elem = Int[]
  for (i, c) in enumerate(cs)
    first_i = length(gen) + 1
    for (k, g) in enumerate(S.G)
      p = _apply(g, c)
      any(j -> sum(abs2, gen[j] .- p) <= tol^2, first_i:length(gen)) && continue
      push!(gen, p); push!(owner, i); push!(elem, k)
    end
  end
  gen, owner, elem
end

# Lloyd's relaxation: each admissible cell goes to the nearest image of a centre, and each centre moves to the
# centroid (weighted by the measures) of its cells brought back by the inverse symmetry; linear per iteration
function _lloyd(cs, G, S, adm, s, tolsnap, maxiter)
  D = G.D
  n = length(cs)
  n == 0 && return cs, 0
  cs = copy(cs)
  it = 0
  tolm = 1e-6 * s
  for outer it in 1:maxiter
    gen, owner, elem = _images(cs, S, tolm)
    B = _Buckets(G.lo, G.hi, s, length(gen))
    foreach(p -> _push!(B, p), gen)
    sx = [ntuple(_ -> 0.0, D) for _ in 1:n]
    sw = zeros(n)
    for c in adm
      j = _nearest_index(B, G.cen[c])
      i = owner[j]
      y = _apply(S.Ginv[elem[j]], G.cen[c])
      sx[i] = sx[i] .+ G.meas[c] .* y
      sw[i] += G.meas[c]
    end
    disp = 0.0
    for i in 1:n
      sw[i] > 0 || continue
      new = _snap(S, sx[i] ./ sw[i], tolsnap)
      disp = max(disp, sqrt(sum(abs2, new .- cs[i])))
      cs[i] = new
    end
    disp < 0.05 * G.hmean && break
  end
  cs, it
end

# Separation (Gauss–Jacobi sweeps, linear per sweep): every two images of different centres closer than dmin are pushed
# apart to dsep > dmin (each half the way, brought back to its centre by the inverse symmetry, averaged over the images
# of the centre); a centre closer than R to a keep-out point is pushed out to R; a centre that leaves the admissible
# cells goes to the nearest admissible centroid. The largest violation of dmin and R left (0 if none).
function _separate!(cs::Vector{NTuple{D,Float64}}, G, S, dmin, dsep, K, R, Cadm, admc, ρp, tolm; sweeps = 60) where {D}
  n = length(cs)
  worst = 0.0
  n == 0 && return worst
  for _ in 1:sweeps
    gen, owner, elem = _images(cs, S, tolm)
    nimg = zeros(Int, n)
    foreach(i -> nimg[i] += 1, owner)
    B = _Buckets(G.lo, G.hi, dsep, length(gen))
    foreach(p -> _push!(B, p), gen)
    disp = [ntuple(_ -> 0.0, D) for _ in 1:n]
    worst = 0.0
    for (j, p) in enumerate(gen)
      i, A = owner[j], S.G[elem[j]].A
      I = _box(B, p)
      for J in CartesianIndices(ntuple(k -> max(1, I[k] - 1):min(B.n[k], I[k] + 1), D))
        q = B.head[_lin(B, Tuple(J))]
        while q != 0
          if owner[q] != i
            d2 = sum(abs2, p .- B.pts[q])
            if d2 < dsep^2
              d = sqrt(d2)
              worst = max(worst, dmin - d)
              u = d > 1e-12 * dsep ? (p .- B.pts[q]) ./ d : ntuple(k -> k == 1 ? (i < owner[q] ? 1.0 : -1.0) : 0.0, D)
              δ = ((dsep - d) / 2) .* u
              disp[i] = disp[i] .+ ntuple(k -> sum(A[l][k] * δ[l] for l in 1:D), D) ./ nimg[i]   # Aᵀδ
            end
          end
          q = B.next[q]
        end
      end
    end
    for i in 1:n
      x = cs[i] .+ disp[i]
      if !isempty(K.pts)
        k = _nearest_index(K, x)
        v = x .- K.pts[k]
        d = sqrt(sum(abs2, v))
        if d < R
          worst = max(worst, R - d)
          x = d > 0 ? K.pts[k] .+ (R * (1 + 1e-3) / d) .* v : x
        end
      end
      _nearest2(Cadm, x) <= ρp^2 || (x = G.cen[admc[_nearest_index(Cadm, x)]])
      cs[i] = _snap(S, x, dmin)
    end
    worst <= 0 && break
  end
  max(worst, 0.0)
end

# adds to pts the candidate centroids farthest from pts and their images until they give N holes (`count`: the holes of
# a point, its distinct images); deterministic; the distances by buckets at the start, then O(candidates × group) per
# added point
function _farthest_fill!(pts, G, S, cand, N, s; count = p -> 1)
  total = sum(count, pts; init = 0)
  (total >= N || isempty(cand)) && return pts
  D = G.D
  imgs = NTuple{D,Float64}[_apply(g, p) for p in pts for g in S.G]
  d2 = fill(Inf, length(cand))
  if !isempty(imgs)
    B = _Buckets(G.lo, G.hi, s, length(imgs))
    foreach(p -> _push!(B, p), imgs)
    for (j, c) in enumerate(cand)
      i = _nearest_index(B, G.cen[c])
      d2[j] = sum(abs2, G.cen[c] .- B.pts[i])
    end
  end
  while total < N
    j = argmax(d2)
    d2[j] > 0 || break
    x = G.cen[cand[j]]
    push!(pts, x)
    total += count(x)
    for g in S.G
      y = _apply(g, x)
      for (k, c) in enumerate(cand)
        d2[k] = min(d2[k], sum(abs2, G.cen[c] .- y))
      end
    end
  end
  pts
end

# points that give N holes (`count`), spread: the first, then each time the one farthest from those taken
# (deterministic)
function _farthest_subset(pts, N; count = p -> 1)
  d2 = fill(Inf, length(pts))
  take = [1]
  total = count(pts[1])
  while total < N
    x = pts[take[end]]
    for j in eachindex(pts)
      d2[j] = min(d2[j], sum(abs2, pts[j] .- x))
    end
    push!(take, argmax(d2))
    total += count(pts[take[end]])
  end
  pts[sort!(take)]
end

# the centres of the holes (see hole_seed); pattern :poisson is internal (random points at least `spacing` apart,
# the random start of :cvt)
function _hole_centers(G, radius, spacing, pattern, margin, exclude, rng, origin, jitter, S)
  D = G.D
  lo, hi = G.lo, G.hi
  L = maximum(hi .- lo)
  R = radius + margin
  keep = _tag_vertices(G.model, exclude, Val(D))           # keep-out points: excluded tags and frozen cells
  for c in findall(G.frozen), p in G.X[c]
    push!(keep, _tup(p, Val(D)))
  end
  K = _Buckets(lo, hi, R, length(keep))
  foreach(p -> _push!(K, p), keep)
  C = _Buckets(lo, hi, radius, length(G.cen))
  foreach(c -> _push!(C, c), G.cen)
  admissible(x) = _nearest2(C, x) <= radius^2 && _nearest2(K, x) >= R^2
  tolF = 1e-9 * L
  fund(x) = _in_fundamental(S, x, tolF)
  # the smallest distance between two centres that the pattern guarantees
  dmin = pattern === :cvt ? 2radius + (spacing - 2radius) / 2 :
         pattern === :poisson ? spacing : 2radius + (1 - jitter) * (spacing - 2radius)
  # the points of the pattern in the fundamental region
  poisson() = begin
    cand = [c for c in 1:length(G.cen) if !G.frozen[c] && _nearest2(K, G.cen[c]) >= R^2 && fund(G.cen[c])]
    for i in length(cand):-1:2                              # Fisher–Yates
      j = rand(rng, 1:i)
      cand[i], cand[j] = cand[j], cand[i]
    end
    B = _Buckets(lo, hi, spacing, ceil(Int, prod(max.(hi .- lo, spacing)) / spacing^D) + 1)
    pts = NTuple{D,Float64}[]
    for c in cand
      x = G.cen[c]
      _nearest2(B, x) >= spacing^2 * (1 - 1e-12) || continue
      _push!(B, x)
      push!(pts, x)
    end
    pts
  end
  pts = if pattern === :poisson
    poisson()
  elseif pattern === :cvt
    # from :hex, the points inside the mesh (within half a cell of a centroid): not those cut by the boundary, which
    # the relaxation would pull inside and crowd
    ρin = G.hmax * sqrt(D) / 2
    Cin = _Buckets(lo, hi, ρin, length(G.cen))
    foreach(c -> _push!(Cin, c), G.cen)
    inside(x) = _nearest2(Cin, x) <= ρin^2
    init = rng === nothing ? filter(x -> fund(x) && admissible(x) && inside(x), _lattice_points(:hex, lo, hi, spacing, origin, radius, Val(D))) :
           poisson()
    adm = [c for c in 1:length(G.cen) if !G.frozen[c] && _nearest2(K, G.cen[c]) >= R^2]
    # the number of holes of :hex with this spacing over the admissible region (√3/2 s² per hole in 2D, s³/√2 in 3D):
    # the start is thinned or completed up to it by farthest points (the boundary layer the start leaves empty); a
    # point gives as many holes as distinct images (those on a plane or an axis, fewer)
    vhex = D == 2 ? sqrt(3) / 2 * spacing^2 : D == 3 ? spacing^3 / sqrt(2) : spacing^D
    N = max(1, round(Int, sum(c -> G.meas[c], adm; init = 0.0) / vhex))
    nimg(p) = length(_orbit(S, _snap(S, p, dmin), 1e-6 * spacing))
    sum(nimg, init; init = 0) > N && (init = _farthest_subset(init, N; count = nimg))
    _farthest_fill!(init, G, S, filter(c -> fund(G.cen[c]), adm), N, spacing; count = nimg)
    n0 = length(init)
    # Lloyd steps, each followed by the separation of the centres closer than dmin (the number of holes is kept: in
    # thin regions, a tube, Lloyd alone packs them closer than spacing); then the last separation
    Cadm = _Buckets(lo, hi, ρin, length(adm))
    foreach(c -> _push!(Cadm, G.cen[c]), adm)
    ρp = min(ρin, radius)
    dsep = dmin * (1 + 1e-2)
    tolm = 1e-6 * spacing
    cs = init
    for _ in 1:60
      prev = copy(cs)
      cs = first(_lloyd(cs, G, S, adm, spacing, dmin, 1))
      _separate!(cs, G, S, dmin, dsep, K, R, Cadm, adm, ρp, tolm)
      maximum(i -> sqrt(sum(abs2, cs[i] .- prev[i])), eachindex(cs); init = 0.0) < 0.05 * G.hmean && break
    end
    worst = _separate!(cs, G, S, dmin, dsep, K, R, Cadm, adm, ρp, tolm; sweeps = 200)
    @debug "hole_seed :cvt" N n0 holes = sum(nimg, cs; init = 0) worst
    cs
  else
    ρj = jitter * (spacing - 2radius) / 2
    r = rng === nothing ? Random.Xoshiro(1) : rng
    pts = filter(fund, _lattice_points(pattern, lo, hi, spacing, origin, radius + ρj, Val(D)))
    if ρj > 0                                               # uniform in the ball of radius ρj (rejection)
      for i in eachindex(pts)
        while true
          δ = ntuple(_ -> 2rand(r) - 1, D)
          sum(abs2, δ) <= 1 || continue
          pts[i] = pts[i] .+ ρj .* δ
          break
        end
      end
    end
    pts
  end
  centers = last(_accept_orbits(pts, S, dmin, admissible, spacing, lo, hi))
  centers, keep
end

# the orbits of the points, accepted whole (symmetric) in order if every point is admissible and at least dmin from
# the others: the points kept (snapped) and all the centres
function _accept_orbits(pts, S, dmin, admissible, spacing, lo::NTuple{D,Float64}, hi) where {D}
  A = _Buckets(lo, hi, dmin, length(pts) * length(S.G))
  d2 = dmin^2 * (1 - 1e-9)
  kept, centers = NTuple{D,Float64}[], NTuple{D,Float64}[]
  for c in pts
    c = _snap(S, c, dmin)
    o = _orbit(S, c, 1e-6 * spacing)
    all(admissible, o) || continue
    all(p -> _nearest2(A, p) >= d2, o) || continue
    all(i -> all(j -> sum(abs2, o[i] .- o[j]) >= d2, i+1:length(o)), eachindex(o)) || continue
    push!(kept, c)
    foreach(p -> (_push!(A, p); push!(centers, p)), o)
  end
  kept, centers
end

# a warning when the mesh or the keep-out points are not symmetric (to the largest cell size)
function _check_symmetry(G, S, keep)
  length(S.G) == 1 && return nothing
  tol2 = G.hmax^2
  Cn = _Buckets(G.lo, G.hi, G.hmax, length(G.cen))
  foreach(c -> _push!(Cn, c), G.cen)
  Kn = _Buckets(G.lo, G.hi, G.hmax, length(keep))
  foreach(c -> _push!(Kn, c), keep)
  for (k, g) in enumerate(S.gens)
    nc = count(x -> _nearest2(Cn, _apply(g, x)) > tol2, G.cen)
    nk = count(x -> _nearest2(Kn, _apply(g, x)) > tol2, keep)
    (nc > 0 || nk > 0) &&
      @warn "hole_seed: symmetry $k does not map the mesh onto itself ($nc of $(length(G.cen)) cell centroids without " *
            "an image) or the frozen zones and excluded tags onto themselves ($nk of $(length(keep)) points): the " *
            "seed is symmetric, the problem is not"
  end
  nothing
end

function _seed_args(V, radius, spacing, pattern, margin, width, jitter, rng, bounds; internal = false)
  (V isa TrialFESpace && num_dirichlet_dofs(V) > 0) &&
    throw(ArgumentError("hole_seed: give the test space (the one given to FieldVariable, with dirichlet_tags)"))
  lb, ub = Float64(bounds[1]), Float64(bounds[2])
  (isfinite(lb) && isfinite(ub) && lb < ub) || throw(ArgumentError("hole_seed: bounds must be finite with lb < ub, got $bounds"))
  (isfinite(radius) && radius > 0) || throw(ArgumentError("hole_seed: radius must be positive, got $radius"))
  (isfinite(width) && width > 0) || throw(ArgumentError("hole_seed: width must be positive, got $width"))
  (isfinite(margin) && margin >= 0) || throw(ArgumentError("hole_seed: margin must be ≥ 0, got $margin"))
  (isfinite(spacing) && spacing > 2radius) ||
    throw(ArgumentError("hole_seed: spacing ($spacing) must be larger than 2 radius ($(2radius)): the holes would overlap"))
  pattern in (:hex, :lattice, :cvt) || (internal && pattern === :poisson) ||
    throw(ArgumentError("hole_seed: pattern must be :hex, :lattice or :cvt, got $(repr(pattern)) (for a random start: " *
                        "pattern = :cvt with rng)"))
  (0 <= jitter < 1) || throw(ArgumentError("hole_seed: jitter must be in [0, 1), got $jitter"))
  (pattern in (:cvt, :poisson) && jitter > 0) &&
    throw(ArgumentError("hole_seed: jitter moves the holes of :hex and :lattice; for a random start use pattern = :cvt with rng"))
  r = rng === nothing ? (pattern === :poisson ? Random.Xoshiro(1) : nothing) :
      rng isa Integer ? Random.Xoshiro(rng) : rng isa Random.AbstractRNG ? rng :
      throw(ArgumentError("hole_seed: rng must be nothing, an integer seed or an AbstractRNG, got $(typeof(rng))"))
  lb, ub, r
end

function _seed_centers(V, G, radius, spacing, pattern, margin, exclude, r, origin, jitter, symmetry)
  ex = _seed_tags(exclude)
  _check_tags(G.model, ex, "exclude")
  origin === nothing || length(origin) == G.D || throw(ArgumentError("hole_seed: origin must have $(G.D) coordinates"))
  S = _symmetry_group(symmetry, Val(G.D), maximum(G.hi .- G.lo))
  centers, keep = _hole_centers(G, Float64(radius), Float64(spacing), pattern, Float64(margin), ex, r, origin, Float64(jitter), S)
  _check_symmetry(G, S, keep)
  centers
end

"""
    hole_centers(V; radius, spacing = 4radius, pattern = :hex, margin = radius, exclude = (), rng = nothing,
                 origin = nothing, jitter = 0.0, symmetry = nothing) -> Vector{NTuple{D,Float64}}

The centres of the holes that [`hole_seed`](@ref) places with the same arguments (not exported; for tests and
plots). `pattern = :poisson` (internal) gives the random points at least `spacing` apart that start `:cvt` with `rng`.
"""
function hole_centers(V; radius::Real, spacing::Real = 4radius, pattern::Symbol = :hex, margin::Real = radius,
                      exclude = (), rng = nothing, origin = nothing, jitter::Real = 0.0, symmetry = nothing)
  _, _, r = _seed_args(V, radius, spacing, pattern, margin, radius, jitter, rng, (-1.0, 1.0); internal = true)
  G = _seed_geometry(V)
  _seed_centers(V, G, radius, spacing, pattern, margin, exclude, r, origin, jitter, symmetry)
end

"""
    hole_seed(V; radius, spacing = 4radius, pattern = :hex, margin = radius, exclude = (), connect = exclude,
              base = nothing, bounds = (-1.0, 1.0), width = radius, frozen = nothing, rng = nothing, origin = nothing,
              jitter = 0.0, symmetry = nothing, verbose = false) -> Vector{Float64}

The free values of a design field with holes: the `init` of [`FieldVariable`](@ref) (same space `V`, same `bounds`,
same `frozen`):

```julia
φ0 = hole_seed(Vφ; radius = 0.05, spacing = 0.2, exclude = "clamp", connect = ("clamp", "load"))
z  = OptVariables(φ = FieldVariable(Vφ; bounds = (-1.0, 1.0), init = φ0, frozen = 1.0))
```

Two uses:
1. A starting design for a method that nucleates holes, such as `ReactionDiffusion(…; sensitivity = :physical)`
   (optional; Yamada et al. 2010 start from perforated designs too): studies of the dependence of the design on the
   start. A solid start is still the default of the examples.
2. The start of a method that only moves the boundary and cannot create holes (the double-well reaction–diffusion
   `DoubleWell`, next release; Takezawa et al. 2010): there the holes of the seed are all the holes the design can
   have, and the seed is mandatory.

Holes of radius `radius` (a length, in the units of the mesh) about `spacing` apart:
- `pattern = :hex` (default): the close-packed lattice with nearest neighbours `spacing` apart, a hole at the centre
  of the bounding box of the mesh: staggered rows `√3/2 spacing` apart in 2D (the seeds of Takezawa et al.),
  face-centred cubic in 3D. `jitter ∈ [0, 1)` moves each centre at random (`rng`, seed 1 if not given) inside a ball
  of radius `jitter (spacing − 2radius)/2`, so that two holes never touch (the solid between them is at least
  `(1 − jitter)(spacing − 2radius)`).
- `pattern = :lattice`: square (cubic) lattice of step `spacing`, symmetric in the bounding box (with `jitter` too).
- `pattern = :cvt`: a centroidal Voronoi tessellation of the cells where a hole may go (Lloyd's relaxation): the
  centres spread evenly over any geometry, away from the excluded zones. The number of holes is that of `:hex` with
  the same `spacing` over that region (its measure over `√3/2 spacing²` in 2D, `spacing³/√2` in 3D), so `spacing` sets
  the density, and neighbours are about `spacing` apart in the bulk. Guaranteed: neighbours at least
  `2radius + (spacing − 2radius)/2` apart (half the solid of `:hex` between two holes). After each Lloyd step the
  centres closer than that are pushed apart, so that the number of holes is kept where Lloyd alone would pack them
  closer (slender parts: a tube a few `spacing` thick); an orbit is dropped only where the minimum cannot be met
  (parts thinner than about `spacing`). It starts from the centres of `:hex` (deterministic) or, with `rng` (an
  integer seed or an `AbstractRNG`), from random points at least `spacing` apart, thinned or completed by farthest
  points to that number; at most 60 Lloyd steps, until no centre moves more than 5 % of the mean cell size. Its
  centres are inside the region (near a centroid of a cell where a hole may go), not cut by the boundary as those of
  `:hex` may be: on a slender part `:hex` keeps more holes than `:cvt`, many of them cut. The sequence of Julia's
  generators may change between versions of Julia: give references without `rng`.

`origin` (`:hex`, `:lattice`, and the start of `:cvt`) is a point of the lattice. A hole is kept when the centroid of
some cell is inside it (meshes that do not fill their bounding box), and holes cut by the boundary are kept (a
periodic pattern); to keep them inside, exclude the boundary tag. The holes keep `radius + margin` from the vertices
of the **frozen zones of `V`** (cells whose values are all Dirichlet values: read from `V`, nothing to give) and of
the tags in `exclude` (loads, supports), so that the solid is at least `margin` thick there.

`symmetry`: one symmetry or a tuple of them, each a named tuple: `(point = p, normal = n)` a reflection in the plane
(line in 2D) through `p` normal to `n`; `(point = p, order = n)` a rotation of order `n` about `p` in 2D and
`(point = p, axis = a, order = n)` about the axis through `p` along `a` in 3D. The seed is built in a fundamental
region of the group they generate and replicated: the centres are exactly symmetric, a centre closer to a plane (or
an axis) than the distance between holes goes onto it, and every replica keeps the distance between holes and the
margins (an orbit that does not is dropped whole). The geometry, the frozen zones and the excluded tags must be
symmetric too: a warning says so when the mesh (to the largest cell size) or the keep-out points are not.
```julia
hole_seed(Vφ; radius = 0.05, spacing = 0.2, symmetry = (point = (1.0, 0.5), normal = (0.0, 1.0)))   # y = 0.5
hole_seed(Vt; radius = 0.1, pattern = :cvt, symmetry = (point = (0, 0, 0), axis = (0, 0, 1), order = 4))
```

The value at a point at distance `d` from the nearest centre goes from `lb` at `d = radius − width` to the interface
level `(lb + ub)/2` at `d = radius` and `ub` at `d = radius + width` (linear in `d`, clamped: a clipped distance for a
level set in `(-1, 1)`, a density in `(0, 1)` with `0.5` on the edge of the holes), and is then the minimum with
`base` (default `ub`; a number, a vector of free values or a function of the coordinates).

Checks: an `ArgumentError` when no hole fits or no value of the field falls inside a hole; warnings when some holes
contain no value of the field (the mesh does not resolve them: refine or enlarge them), when the solid between two
holes is thinner than two cells, and when the solid (the cells whose mean value is at least the level, the frozen
cells with the values `frozen`, by default `ub`) does not join the tags of `connect` (by default `exclude`;
`nothing` skips the check) through faces. `verbose = true` prints the number of holes and the fraction of values in
them.

Void fraction of a periodic pattern with `spacing = 4radius`: `:hex` 23 % in 2D and 9 % in 3D, `:lattice` 20 % and
7 %; in 2D the best designs measured came from about four rows of holes across the smaller side of the domain
(`spacing ≈ L_min/4`). A pattern of cosines (Oka, Misawa, Yamada) is one line of the user,
`init = x -> clamp(-cos(2π x[1]/s) * cos(2π x[2]/s) + c, -1, 1)`, without the margins.

P0 and P1 Lagrangian spaces (and other scalar Lagrangian spaces: the values sit at the nodes), any mesh in 2D and 3D,
with periodic conditions too; the cost is linear in the cells and the values (a few microseconds per value; `:cvt`
that times its Lloyd steps). Sequential (not for distributed spaces yet).
"""
function hole_seed(V; radius::Real, spacing::Real = 4radius, pattern::Symbol = :hex, margin::Real = radius,
                   exclude = (), connect = exclude, base = nothing, bounds = (-1.0, 1.0), width::Real = radius,
                   frozen = nothing, rng = nothing, origin = nothing, jitter::Real = 0.0, symmetry = nothing,
                   verbose::Bool = false)
  lb, ub, r = _seed_args(V, radius, spacing, pattern, margin, width, jitter, rng, bounds)
  radius, spacing, margin, width, jitter = Float64.((radius, spacing, margin, width, jitter))
  G = _seed_geometry(V)
  D = G.D
  _check_tags(G.model, _seed_tags(exclude), "exclude")
  co = _seed_tags(connect)
  _check_tags(G.model, co, "connect")
  hmean = G.hmean
  ligament = pattern === :cvt ? (spacing - 2radius) / 2 : (1 - jitter) * (spacing - 2radius)
  ligament >= 2hmean ||
    @warn "hole_seed: the solid between two holes ($(round(ligament, sigdigits = 3))) is thinner than two cells (mean cell size $(round(hmean, sigdigits = 3)))"
  centers = _seed_centers(V, G, radius, spacing, pattern, margin, exclude, r, origin, jitter, symmetry)
  isempty(centers) &&
    throw(ArgumentError("hole_seed: no hole fits (a centroid of a cell inside each hole, radius + margin = $(radius + margin) " *
                        "from the frozen zones and the excluded tags)"))
  # the coordinates of the free values by interpolation: P0, P1, any numbering
  n = num_free_dofs(V)
  xs = ntuple(k -> get_free_dof_values(interpolate(x -> x[k], V)), D)
  B = _Buckets(G.lo, G.hi, radius + width, length(centers))
  foreach(c -> _push!(B, c), centers)
  bv = base === nothing ? nothing : base isa Number ? nothing : base isa AbstractVector ? base :
       get_free_dof_values(interpolate(base, V))
  bv === nothing || length(bv) == n ||
    throw(DimensionMismatch("hole_seed: base has $(length(bv)) values, the space has $n free values"))
  b0 = base isa Number ? Float64(base) : ub
  φ = Vector{Float64}(undef, n)
  @inbounds for i in 1:n
    x = ntuple(k -> xs[k][i], D)
    d2 = _nearest2(B, x)
    s = d2 == Inf ? 1.0 : clamp((sqrt(d2) - radius) / width, -1.0, 1.0)
    v = lb + (ub - lb) * (s + 1) / 2
    φ[i] = clamp(min(bv === nothing ? b0 : Float64(bv[i]), v), lb, ub)
  end
  level = (lb + ub) / 2
  # holes without any value of the field inside
  E = _Buckets(G.lo, G.hi, radius, n)
  nin = 0
  for i in 1:n
    x = ntuple(k -> xs[k][i], D)
    _nearest2(B, x) < radius^2 || continue
    nin += 1
    _push!(E, x)
  end
  nin > 0 || throw(ArgumentError("hole_seed: no value of the field falls inside a hole (radius = $radius, mean cell " *
                                 "size $(round(hmean, sigdigits = 3))): refine the mesh or enlarge the holes"))
  empty = count(c -> _nearest2(E, c) >= radius^2, centers)
  empty > 0 &&
    @warn "hole_seed: $empty of $(length(centers)) holes contain no value of the field (radius = $radius, mean cell size $(round(hmean, sigdigits = 3))): refine the mesh or enlarge the holes"
  connected = true
  if !isempty(co)
    connected, why = _solid_connects(V, G, φ, level, frozen === nothing ? ub : frozen, co)
    connected || @warn "hole_seed: the solid of the seed does not join the tags $(join(co, ", ")): $why"
  end
  verbose && @info "hole_seed: $(length(centers)) holes ($pattern), $(round(100count(<(level), φ) / n, digits = 1)) % of " *
                   "the values below the level" * (isempty(co) ? "" : connected ? ", connected" : ", not connected")
  φ
end

# The cells whose mean value is ≥ level are solid; two solid cells are joined if they share a face of dimension D − 1
# (not a vertex: a hinge). The tags are joined if one component of the solid touches all of them.
function _solid_connects(V, G, φ, level, frozen, tags)
  D = G.D
  topo = Gridap.Geometry.get_grid_topology(G.model)
  ncm = Gridap.Geometry.num_faces(topo, D)
  t2m = try
    Gridap.Geometry.get_glue(G.Ω, Val(D)).tface_to_mface
  catch
    nothing
  end
  t2m === nothing && return true, ""        # not reached for the triangulations of a model and of its parts
  dv = (frozen isa Number || num_dirichlet_dofs(V) == 0) ? nothing : get_dirichlet_dof_values(TrialFESpace(V, frozen))
  solid = falses(ncm)
  for (c, cell) in enumerate(G.ids)
    s = 0.0
    for d in cell
      s += d > 0 ? φ[d] : dv === nothing ? Float64(frozen) : dv[-d]
    end
    solid[t2m[c]] = s / length(cell) >= level
  end
  c2f = Gridap.Geometry.get_faces(topo, D, D - 1)
  f2c = Gridap.Geometry.get_faces(topo, D - 1, D)
  comp = zeros(Int, ncm)
  ncomp = 0
  stack = Int[]
  for c0 in 1:ncm
    (solid[c0] && comp[c0] == 0) || continue
    ncomp += 1
    comp[c0] = ncomp
    push!(stack, c0)
    while !isempty(stack)
      c = pop!(stack)
      for f in c2f[c], c2 in f2c[f]
        (solid[c2] && comp[c2] == 0) || continue
        comp[c2] = ncomp
        push!(stack, c2)
      end
    end
  end
  labels = get_face_labeling(G.model)
  common = Set{Int}()
  for (i, t) in enumerate(tags)
    touched = Set{Int}()
    for d in 0:D
      m = Gridap.Geometry.get_face_mask(labels, t, d)
      any(m) || continue
      if d == D
        for f in findall(m)
          comp[f] > 0 && push!(touched, comp[f])
        end
      else
        f2cells = Gridap.Geometry.get_faces(topo, d, D)
        for f in findall(m), c in f2cells[f]
          comp[c] > 0 && push!(touched, comp[c])
        end
      end
    end
    isempty(touched) && return false, "no solid cell at the tag $t"
    common = i == 1 ? touched : intersect(common, touched)
    isempty(common) && return false, "the tag $t is in another piece of the solid"
  end
  true, ""
end
