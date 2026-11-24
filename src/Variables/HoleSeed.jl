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
  n = ntuple(k -> max(1, ceil(Int, (hi[k] - lo[k]) / side)), Val(D))
  _Buckets{D}(side, lo, n, zeros(Int, prod(n)), Int[], NTuple{D,Float64}[])
end

@inline _box(b::_Buckets{D}, x) where {D} = ntuple(k -> clamp(floor(Int, (x[k] - b.lo[k]) / b.side) + 1, 1, b.n[k]), Val(D))

# the boxes within m of the box I (a function, so that loops that rebuild the buckets do not box them in a closure)
@inline _around(b::_Buckets{D}, I, m) where {D} = CartesianIndices(ntuple(k -> max(1, I[k] - m):min(b.n[k], I[k] + m), Val(D)))

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
  for J in _around(b, I, 1)
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
    for J in _around(b, I, ring)
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

_tup(p, ::Val{D}) where {D} = ntuple(k -> Float64(p[k]), Val(D))
_dimval(::Val{D}) where {D} = D          # D as a constant from the type of G.dim (no dynamic ntuple)
# arithmetic on points without broadcasting (nested tuple broadcasts allocate when they are not inlined)
@inline _d2(a::NTuple{D,Float64}, b::NTuple{D,Float64}) where {D} = (s = 0.0; for k in 1:D; s += (a[k] - b[k])^2; end; s)
@inline _add(a::NTuple{D,Float64}, b::NTuple{D,Float64}) where {D} = ntuple(k -> a[k] + b[k], Val(D))
@inline _sub(a::NTuple{D,Float64}, b::NTuple{D,Float64}) where {D} = ntuple(k -> a[k] - b[k], Val(D))
@inline _axpy(α, x::NTuple{D,Float64}, y::NTuple{D,Float64}) where {D} = ntuple(k -> y[k] + α * x[k], Val(D))   # y + αx
@inline _scal(α, x::NTuple{D,Float64}) where {D} = ntuple(k -> α * x[k], Val(D))
@inline _div(x::NTuple{D,Float64}, α) where {D} = ntuple(k -> x[k] / α, Val(D))

# the indices of the points within distance r of x (r ≤ the side of the boxes searches 3^D boxes; Inf, all of them)
@inline function _foreach_within(f, b::_Buckets{D}, x, r) where {D}
  I = _box(b, x)
  m = isfinite(r) ? max(1, ceil(Int, r / b.side)) : maximum(b.n)
  r2 = r^2
  for J in _around(b, I, m)
    p = b.head[_lin(b, Tuple(J))]
    while p != 0
      q = b.pts[p]
      d2 = 0.0
      for k in 1:D
        d2 += (x[k] - q[k])^2
      end
      d2 < r2 && f(p, d2)
      p = b.next[p]
    end
  end
  nothing
end

# a max-heap of (value, index) with stale entries (the caller skips those whose value is no longer current)
function _heap_push!(v::Vector{Float64}, ix::Vector{Int}, val, i)
  push!(v, val); push!(ix, i)
  j = length(v)
  while j > 1 && v[j >> 1] < v[j]
    k = j >> 1
    v[j], v[k] = v[k], v[j]; ix[j], ix[k] = ix[k], ix[j]
    j = k
  end
end
function _heap_pop!(v::Vector{Float64}, ix::Vector{Int})
  top = (v[1], ix[1])
  v[1], ix[1] = v[end], ix[end]
  pop!(v); pop!(ix)
  j, n = 1, length(v)
  while true
    l, r, m = 2j, 2j + 1, j
    l <= n && v[l] > v[m] && (m = l)
    r <= n && v[r] > v[m] && (m = r)
    m == j && break
    v[j], v[m] = v[m], v[j]; ix[j], ix[m] = ix[m], ix[j]
    j = m
  end
  top
end

_seed_tags(t) = t === nothing ? () : t isa Union{AbstractString,Symbol} ? (String(t),) : Tuple(String(s) for s in t)

# ---------------------------------------------------------------------------------------------------------------
# Symmetries: isometries x ↦ A x + b (A orthogonal), the finite group they generate

struct _Isometry{D}
  A::NTuple{D,NTuple{D,Float64}}      # rows
  b::NTuple{D,Float64}
end

@inline _apply(g::_Isometry{D}, x) where {D} = ntuple(k -> g.b[k] + sum(g.A[k][j] * x[j] for j in 1:D), Val(D))
_compose(g::_Isometry{D}, h::_Isometry{D}) where {D} =        # g ∘ h
  _Isometry{D}(ntuple(i -> ntuple(j -> sum(g.A[i][k] * h.A[k][j] for k in 1:D), Val(D)), Val(D)), _apply(g, h.b))
function _inverse(g::_Isometry{D}) where {D}
  At = ntuple(i -> ntuple(j -> g.A[j][i], Val(D)), Val(D))
  _Isometry{D}(At, ntuple(i -> -sum(At[i][k] * g.b[k] for k in 1:D), Val(D)))
end
_identity(::Val{D}) where {D} = _Isometry{D}(ntuple(i -> ntuple(j -> Float64(i == j), Val(D)), Val(D)), ntuple(_ -> 0.0, Val(D)))
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
    A = ntuple(i -> ntuple(j -> Float64(i == j) - 2n[i] * n[j], Val(D)), Val(D))
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
  _Isometry{D}(A, ntuple(i -> p[i] - sum(A[i][j] * p[j] for j in 1:D), Val(D)))
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
  # w must not be fixed by the linear part of any element (a plane that contains w, an axis along w): the first of a
  # few generic directions that every element moves by at least 0.1, or the one moved most
  cands = D == 2 ? ((0.8320502943378437, 0.5547001962252291), (0.5547001962252291, -0.8320502943378437),
                    (0.9805806756909202, 0.19611613513818404)) :
                   ((0.8017837257372732, 0.5345224838248488, 0.2672612419124244),
                    (0.2672612419124244, -0.8017837257372732, 0.5345224838248488),
                    (0.5345224838248488, 0.2672612419124244, -0.8017837257372732),
                    (0.9128709291752769, -0.3651483716701107, 0.18257418583505536))
  moved(w) = minimum((sqrt(sum(abs2, ntuple(i -> sum(g.A[i][j] * w[j] for j in 1:D) - w[i], Val(D)))) for g in els[2:end]); init = Inf)
  k = findfirst(w -> moved(w) >= 0.1, cands)
  w = k === nothing ? cands[argmax(map(moved, cands))] : cands[k]
  (; G = els, Ginv = map(_inverse, els), gens = gs, w = ntuple(k -> Float64(w[k]), Val(D)), tol)
end

@inline _dotw(S, x) = sum(S.w .* x)
_in_fundamental(S, x, tol) = (wx = _dotw(S, x); all(g -> _dotw(S, _apply(g, x)) <= wx + tol, S.G))

# the distinct images of c (points closer than tol are one)
function _orbit(S, c, tol)
  o = [c]
  for g in S.G
    p = _apply(g, c)
    any(q -> _d2(p, q) <= tol^2, o) || push!(o, p)
  end
  o
end

# the subgroup generated by the elements H (indices into S.G): the closure of the identity under left products
function _subgroup(S, H)
  sub = [1]
  i = 1
  while i <= length(sub)
    for h in H
      e = _compose(S.G[h], S.G[sub[i]])
      k = findfirst(g -> _close(g, e, S.tol), S.G)
      k === nothing || k in sub || push!(sub, k)
    end
    i += 1
  end
  sub
end

# the centre moved onto the plane or the axis it is near to: the mean of its images by the subgroup generated by the
# elements that move it less than d (the projection onto the fixed set of that subgroup), repeated while that subgroup
# grows. Images exactly d apart (neighbours of a lattice) do not count: relative tolerance 1e-9
function _snap(S, c, d)
  length(S.G) == 1 && return c
  d2 = d^2 * (1 - 1e-9)
  prev = Int[]
  for _ in 1:4
    H = Int[]
    for k in 2:length(S.G)
      _d2(_apply(S.G[k], c), c) < d2 && push!(H, k)
    end
    isempty(H) && return c
    sub = sort!(_subgroup(S, H))
    sub == prev && return c
    s = _scal(0.0, c)
    for k in sub
      s = _add(s, _apply(S.G[k], c))
    end
    c = _div(s, length(sub))
    prev = sub
  end
  c
end

# ---------------------------------------------------------------------------------------------------------------
# What the seed reads from the space: cell coordinates, centroids, sizes and measures, the frozen cells (all their
# values are Dirichlet values, as for FieldVariable), the box of the mesh
function _seed_geometry(V)
  Ω = get_triangulation(V)
  num_point_dims(Ω) == num_cell_dims(Ω) ||
    throw(ArgumentError("hole_seed: the cells must have the dimension of the space (not shells or curves in 3D)"))
  _seed_geometry(V, Ω, Val(num_cell_dims(Ω)))
end

function _seed_geometry(V, Ω, ::Val{D}) where {D}
  model = get_background_model(Ω)
  # the nodes of the cells as plain arrays (the lazy cell coordinates allocate at each access)
  nodes = Gridap.Geometry.get_node_coordinates(Ω)
  xn = Vector{NTuple{D,Float64}}(undef, length(nodes))
  for (i, p) in enumerate(nodes)
    xn[i] = _tup(p, Val(D))
  end
  X = Gridap.Arrays.Table(Gridap.Geometry.get_cell_node_ids(Ω))
  nc = length(X)
  cen = Vector{NTuple{D,Float64}}(undef, nc)
  h = Vector{Float64}(undef, nc)
  lo, hi = ntuple(_ -> Inf, Val(D)), ntuple(_ -> -Inf, Val(D))
  @inbounds for c in 1:nc
    r = X.ptrs[c]:X.ptrs[c+1]-1
    sc, m = ntuple(_ -> 0.0, Val(D)), 0.0
    for a in r
      p = xn[X.data[a]]
      sc = _add(sc, p)
      lo, hi = min.(lo, p), max.(hi, p)
      for b in a+1:last(r)
        m = max(m, sqrt(_d2(p, xn[X.data[b]])))
      end
    end
    cen[c] = _div(sc, length(r))
    # the longest distance between two vertices; the diagonal of an n-cube divided by √D
    h[c] = length(r) == 2^D && D > 1 ? m / sqrt(D) : m
  end
  dΩ = get_cell_measure(Ω)
  cm = Gridap.Arrays.array_cache(dΩ)
  meas = Float64[Gridap.Arrays.getindex!(cm, dΩ, c) for c in 1:nc]
  ids = get_cell_dof_ids(V)
  ci = Gridap.Arrays.array_cache(ids)
  frozen = BitVector(undef, nc)
  for c in 1:nc
    frozen[c] = all(<(0), Gridap.Arrays.getindex!(ci, ids, c))
  end
  (; Ω, model, D, dim = Val(D), X, xn, cen, h, meas, frozen, ids, lo, hi, hmean = sum(h) / nc, hmax = maximum(h))
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
      pattern === :hex ? ntuple(k -> (lo[k] + hi[k]) / 2, Val(D)) :
      ntuple(k -> (lo[k] + hi[k]) / 2 - (max(1, round(Int, (hi[k] - lo[k]) / s)) - 1) / 2 * s, Val(D))
  cell, basis = if pattern === :lattice || D ∉ (2, 3)
    ntuple(_ -> s, Val(D)), (ntuple(_ -> 0.0, Val(D)),)
  elseif D == 2
    (s, sqrt(3) * s), ((0.0, 0.0), (s / 2, sqrt(3) * s / 2))
  else
    a = sqrt(2) * s
    (a, a, a), ((0.0, 0.0, 0.0), (a / 2, a / 2, 0.0), (a / 2, 0.0, a / 2), (0.0, a / 2, a / 2))
  end
  ranges = ntuple(k -> (floor(Int, (lo[k] - ext - o[k]) / cell[k]) - 1):(ceil(Int, (hi[k] + ext - o[k]) / cell[k]) + 1), Val(D))
  pts = NTuple{D,Float64}[]
  for I in CartesianIndices(ranges), b in basis
    push!(pts, ntuple(k -> o[k] + I[k] * cell[k] + b[k], Val(D)))
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
      any(j -> _d2(gen[j], p) <= tol^2, first_i:length(gen)) && continue
      push!(gen, p); push!(owner, i); push!(elem, k)
    end
  end
  gen, owner, elem
end

_maxmove(a, b) = (m = 0.0; for i in eachindex(a); m = max(m, sqrt(_d2(a[i], b[i]))); end; m)

# the mean distance from each point to the nearest other point (at most 3s)
function _mean_nn(pts::Vector{NTuple{D,Float64}}, lo, hi, s) where {D}
  length(pts) < 2 && return 3s
  B = _Buckets(lo, hi, s, length(pts))
  for p in pts; _push!(B, p); end
  tot = 0.0
  best = Ref(Inf)
  for (i, p) in enumerate(pts)
    best[] = 9s^2
    _foreach_within(B, p, 3s) do j, d2
      j != i && d2 < best[] && (best[] = d2)
    end
    tot += sqrt(best[])
  end
  tot / length(pts)
end

# Lloyd's relaxation: each admissible cell goes to the nearest image of a centre, and each centre moves to the
# centroid (weighted by the measures) of its cells brought back by the inverse symmetry; linear per iteration
function _lloyd(cs, G, S, adm, s, tolsnap, maxiter)
  D = _dimval(G.dim)
  n = length(cs)
  n == 0 && return cs, 0
  cs = copy(cs)
  it = 0
  tolm = 1e-6 * s
  for outer it in 1:maxiter
    gen, owner, elem = _images(cs, S, tolm)
    B = _Buckets(G.lo, G.hi, s, length(gen))
    for p in gen; _push!(B, p); end
    sx = fill(ntuple(_ -> 0.0, Val(D)), n)
    sw = zeros(n)
    for c in adm
      j = _nearest_index(B, G.cen[c])
      i = owner[j]
      y = _apply(S.Ginv[elem[j]], G.cen[c])
      sx[i] = _axpy(G.meas[c], y, sx[i])
      sw[i] += G.meas[c]
    end
    disp = 0.0
    for i in 1:n
      sw[i] > 0 || continue
      new = _snap(S, _div(sx[i], sw[i]), tolsnap)
      disp = max(disp, sqrt(_d2(new, cs[i])))
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
    for i in owner; nimg[i] += 1; end
    B = _Buckets(G.lo, G.hi, dsep, length(gen))
    for p in gen; _push!(B, p); end
    disp = fill(ntuple(_ -> 0.0, Val(D)), n)
    worst = 0.0
    for (j, p) in enumerate(gen)
      i, A = owner[j], S.G[elem[j]].A
      I = _box(B, p)
      for J in _around(B, I, 1)
        q = B.head[_lin(B, Tuple(J))]
        while q != 0
          if owner[q] != i
            d2 = _d2(p, B.pts[q])
            if d2 < dsep^2
              d = sqrt(d2)
              worst = max(worst, dmin - d)
              sg = i < owner[q] ? 1.0 : -1.0                 # coincident images: apart along the first axis
              u = d > 1e-12 * dsep ? _div(_sub(p, B.pts[q]), d) : ntuple(k -> k == 1 ? sg : 0.0, Val(D))
              δ = _scal((dsep - d) / 2, u)
              disp[i] = _add(disp[i], _div(ntuple(k -> sum(A[l][k] * δ[l] for l in 1:D), Val(D)), nimg[i]))   # Aᵀδ
            end
          end
          q = B.next[q]
        end
      end
    end
    for i in 1:n
      x = _add(cs[i], disp[i])
      if !isempty(K.pts)
        k = _nearest_index(K, x)
        v = _sub(x, K.pts[k])
        d = sqrt(sum(abs2, v))
        if d < R
          worst = max(worst, R - d)
          x = d > 0 ? _axpy(R * (1 + 1e-3) / d, v, K.pts[k]) : x
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
# a point, its distinct images); deterministic. A max-heap of the distances; a new point only lowers the distances of
# the candidates nearer to it than the current largest distance (searched by buckets): linear in the candidates when
# the points are spread over the region
function _farthest_fill!(pts, G, S, cand, N, s; count = p -> 1)
  total = sum(count, pts; init = 0)
  (total >= N || isempty(cand)) && return pts
  D = _dimval(G.dim)
  imgs = NTuple{D,Float64}[_apply(g, p) for p in pts for g in S.G]
  d2 = fill(Inf, length(cand))
  if !isempty(imgs)
    B = _Buckets(G.lo, G.hi, s, length(imgs))
    for p in imgs; _push!(B, p); end
    for (j, c) in enumerate(cand)
      i = _nearest_index(B, G.cen[c])
      d2[j] = _d2(G.cen[c], B.pts[i])
    end
  end
  Bc = _Buckets(G.lo, G.hi, s / 4, length(cand))
  for c in cand; _push!(Bc, G.cen[c]); end
  hv, hi_ = Float64[], Int[]
  for j in eachindex(cand)                                   # in order: ties go to the first candidate
    _heap_push!(hv, hi_, d2[j], j)
  end
  while total < N && !isempty(hv)
    v, j = _heap_pop!(hv, hi_)
    v == d2[j] || continue                                   # stale
    v > 0 || break
    x = G.cen[cand[j]]
    push!(pts, x)
    total += count(x)
    for g in S.G
      y = _apply(g, x)
      _foreach_within(Bc, y, sqrt(v)) do k, e2
        e2 < d2[k] || return
        d2[k] = e2
        _heap_push!(hv, hi_, e2, k)
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
      d2[j] = min(d2[j], _d2(pts[j], x))
    end
    push!(take, argmax(d2))
    total += count(pts[take[end]])
  end
  pts[sort!(take)]
end

# the centres of the holes (see hole_seed); pattern :poisson is internal (random points at least `spacing` apart,
# the random start of :cvt)
function _hole_centers(G, radius, spacing, pattern, margin, exclude, rng, origin, jitter, S, nholes, min_distance)
  D = _dimval(G.dim)
  lo, hi = G.lo, G.hi
  L = maximum(hi .- lo)
  R = radius + margin
  keep = _tag_vertices(G.model, exclude, Val(D))           # keep-out points: excluded tags and frozen cells
  for c in findall(G.frozen), a in G.X.ptrs[c]:G.X.ptrs[c+1]-1
    push!(keep, G.xn[G.X.data[a]])
  end
  K = _Buckets(lo, hi, R, length(keep))
  for p in keep; _push!(K, p); end
  C = _Buckets(lo, hi, radius, length(G.cen))
  for c in G.cen; _push!(C, c); end
  admissible(x) = _nearest2(C, x) <= radius^2 && _nearest2(K, x) >= R^2
  tolF = 1e-9 * L
  fund(x) = _in_fundamental(S, x, tolF)
  # the smallest distance between two centres that the pattern guarantees
  dmin = pattern === :cvt ? (min_distance === nothing ? 2radius + (spacing - 2radius) / 2 : Float64(min_distance)) :
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
    for c in G.cen; _push!(Cin, c); end
    inside(x) = _nearest2(Cin, x) <= ρin^2
    init = rng === nothing ? filter(x -> fund(x) && admissible(x) && inside(x), _lattice_points(:hex, lo, hi, spacing, origin, radius, Val(D))) :
           poisson()
    adm = [c for c in 1:length(G.cen) if !G.frozen[c] && _nearest2(K, G.cen[c]) >= R^2]
    nimg(p) = length(_orbit(S, _snap(S, p, dmin), 1e-6 * spacing))
    candf = filter(c -> fund(G.cen[c]), adm)
    Cadm = _Buckets(lo, hi, ρin, length(adm))
    for c in adm; _push!(Cadm, G.cen[c]); end
    ρp = min(ρin, radius)
    dsep = dmin * (1 + 1e-2)
    tolm = 1e-6 * spacing
    # N holes: the start thinned or completed by farthest points (the boundary layer the start leaves empty; a point
    # gives as many holes as distinct images); Lloyd steps, each followed by the separation of the centres closer than
    # dmin (Lloyd alone packs them closer in thin regions, a tube); then the last separation
    relax(N) = begin
      cs = copy(init)
      sum(nimg, cs; init = 0) > N && (cs = _farthest_subset(cs, N; count = nimg))
      _farthest_fill!(cs, G, S, candf, N, spacing; count = nimg)
      for _ in 1:60
        prev = copy(cs)
        cs = first(_lloyd(cs, G, S, adm, spacing, dmin, 1))
        _separate!(cs, G, S, dmin, dsep, K, R, Cadm, adm, ρp, tolm)
        _maxmove(cs, prev) < 0.05 * G.hmean && break
      end
      _separate!(cs, G, S, dmin, dsep, K, R, Cadm, adm, ρp, tolm; sweeps = 200)
      cs, _mean_nn(first(_images(cs, S, tolm)), lo, hi, spacing)
    end
    if nholes !== nothing
      first(relax(nholes))
    else
      # the number of holes for a mean distance to the nearest neighbour ≈ spacing: from the number of :hex over the
      # admissible region (√3/2 s² per hole in 2D, s³/√2 in 3D), corrected with d̄ ∝ N^(-1/α): α = D at first, then
      # α measured from the two runs (between 1, a slender part, and D)
      vhex = D == 2 ? sqrt(3) / 2 * spacing^2 : D == 3 ? spacing^3 / sqrt(2) : spacing^D
      Ns = [max(1, round(Int, sum(c -> G.meas[c], adm; init = 0.0) / vhex))]
      runs = [relax(Ns[1])]
      for pass in 1:2
        d = last(runs[end])
        abs(d / spacing - 1) <= 0.05 && break
        α = pass == 1 || last(runs[1]) == last(runs[2]) || Ns[1] == Ns[2] ? Float64(D) :
            clamp(log(Ns[2] / Ns[1]) / log(last(runs[1]) / last(runs[2])), 1.0, Float64(D))
        Nn = max(1, round(Int, Ns[end] * (d / spacing)^α))
        Nn in Ns && break
        push!(Ns, Nn)
        push!(runs, relax(Nn))
      end
      k = argmin([abs(last(q) / spacing - 1) for q in runs])
      @debug "hole_seed :cvt" Ns dmean = last.(runs) chosen = k
      first(runs[k])
    end
  else
    ρj = jitter * (spacing - 2radius) / 2
    r = rng === nothing ? Random.Xoshiro(1) : rng
    pts = filter(fund, _lattice_points(pattern, lo, hi, spacing, origin, radius + ρj, Val(D)))
    if ρj > 0                                               # uniform in the ball of radius ρj (rejection)
      for i in eachindex(pts)
        while true
          δ = ntuple(_ -> 2rand(r) - 1, Val(D))
          sum(abs2, δ) <= 1 || continue
          pts[i] = _axpy(ρj, δ, pts[i])
          break
        end
      end
    end
    pts
  end
  _, centers, ndrop = _accept_orbits(pts, S, dmin, admissible, spacing, lo, hi)
  pattern in (:hex, :lattice) && _check_lattice(pattern, S, lo, hi, spacing, origin, radius, ndrop, Val(D))
  centers, keep
end

# a warning when a symmetry does not map the lattice of :hex or :lattice onto itself (inside the box): the replicas
# of the fundamental region then meet along seams where orbits closer than the distance between holes are dropped
function _check_lattice(pattern, S, lo, hi, spacing, origin, radius, ndrop, ::Val{D}) where {D}
  length(S.G) == 1 && return nothing
  L = _lattice_points(pattern, lo, hi, spacing, origin, radius, Val(D))
  B = _Buckets(lo, hi, spacing, length(L))
  for p in L; _push!(B, p); end
  tol2 = (1e-6 * spacing)^2
  inbox(x) = all(k -> lo[k] - 1e-9spacing <= x[k] <= hi[k] + 1e-9spacing, 1:D)
  for (k, g) in enumerate(S.gens)
    bad = count(p -> inbox(p) && inbox(_apply(g, p)) && _nearest2(B, _apply(g, p)) > tol2, L)
    bad > 0 &&
      @warn "hole_seed: symmetry $k does not map the lattice of :$pattern onto itself ($bad lattice points without an " *
            "image): the replicas meet along seams, where $ndrop orbits closer than the distance between holes were " *
            "dropped and gaps are left. Choose origin so that the symmetry maps the lattice onto itself, or use " *
            "pattern = :cvt"
  end
  nothing
end

# the orbits of the points, accepted whole (symmetric) in order if every point is admissible and at least dmin from
# the others: the points kept (snapped), all the centres and the number of admissible orbits dropped for distance
function _accept_orbits(pts, S, dmin, admissible, spacing, lo::NTuple{D,Float64}, hi) where {D}
  A = _Buckets(lo, hi, dmin, length(pts) * length(S.G))
  d2 = dmin^2 * (1 - 1e-9)
  kept, centers = NTuple{D,Float64}[], NTuple{D,Float64}[]
  ndrop = 0
  for c in pts
    c = _snap(S, c, dmin)
    o = _orbit(S, c, 1e-6 * spacing)
    all(admissible, o) || continue
    if !all(p -> _nearest2(A, p) >= d2, o) || !all(i -> all(j -> _d2(o[i], o[j]) >= d2, i+1:length(o)), eachindex(o))
      # a duplicate of an accepted orbit (a point on the border of the fundamental region) is not a drop
      any(p -> _nearest2(A, p) <= (1e-6 * spacing)^2, o) || (ndrop += 1)
      continue
    end
    push!(kept, c)
    foreach(p -> (_push!(A, p); push!(centers, p)), o)
  end
  kept, centers, ndrop
end

# a warning when the mesh or the keep-out points are not symmetric (to the largest cell size)
function _check_symmetry(G, S, keep)
  length(S.G) == 1 && return nothing
  tol2 = G.hmax^2
  Cn = _Buckets(G.lo, G.hi, G.hmax, length(G.cen))
  for c in G.cen; _push!(Cn, c); end
  Kn = _Buckets(G.lo, G.hi, G.hmax, length(keep))
  for c in keep; _push!(Kn, c); end
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

function _seed_args(V, radius, spacing, pattern, margin, width, jitter, rng, bounds, nholes, min_distance; internal = false)
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
  (nholes === nothing && min_distance === nothing) || pattern === :cvt ||
    throw(ArgumentError("hole_seed: nholes and min_distance are options of pattern = :cvt (the lattices fix both)"))
  nholes === nothing || (nholes isa Integer && nholes >= 1) ||
    throw(ArgumentError("hole_seed: nholes must be a positive integer, got $nholes"))
  min_distance === nothing || (isfinite(min_distance) && min_distance > 2radius) ||
    throw(ArgumentError("hole_seed: min_distance ($min_distance) must be larger than 2 radius ($(2radius)): the holes would overlap"))
  r = rng === nothing ? (pattern === :poisson ? Random.Xoshiro(1) : nothing) :
      rng isa Integer ? Random.Xoshiro(rng) : rng isa Random.AbstractRNG ? rng :
      throw(ArgumentError("hole_seed: rng must be nothing, an integer seed or an AbstractRNG, got $(typeof(rng))"))
  lb, ub, r
end

function _seed_centers(G, radius, spacing, pattern, margin, exclude, r, origin, jitter, symmetry, nholes, min_distance)
  ex = _seed_tags(exclude)
  _check_tags(G.model, ex, "exclude")
  origin === nothing || length(origin) == G.D || throw(ArgumentError("hole_seed: origin must have $(G.D) coordinates"))
  S = _symmetry_group(symmetry, Val(G.D), maximum(G.hi .- G.lo))
  centers, keep = _hole_centers(G, Float64(radius), Float64(spacing), pattern, Float64(margin), ex, r, origin, Float64(jitter), S,
                                nholes, min_distance)
  _check_symmetry(G, S, keep)
  centers
end

"""
    hole_centers(V; radius, spacing = 4radius, pattern = :hex, margin = radius, exclude = (), rng = nothing,
                 origin = nothing, jitter = 0.0, symmetry = nothing, nholes = nothing,
                 min_distance = nothing) -> Vector{NTuple{D,Float64}}

The centres of the holes that [`hole_seed`](@ref) places with the same arguments (not exported; for tests and
plots). `pattern = :poisson` (internal) gives the random points at least `spacing` apart that start `:cvt` with `rng`.
"""
function hole_centers(V; radius::Real, spacing::Real = 4radius, pattern::Symbol = :hex, margin::Real = radius,
                      exclude = (), rng = nothing, origin = nothing, jitter::Real = 0.0, symmetry = nothing,
                      nholes = nothing, min_distance = nothing)
  _, _, r = _seed_args(V, radius, spacing, pattern, margin, radius, jitter, rng, (-1.0, 1.0), nholes, min_distance; internal = true)
  G = _seed_geometry(V)
  _seed_centers(G, radius, spacing, pattern, margin, exclude, r, origin, jitter, symmetry, nholes, min_distance)
end

"""
    hole_seed(V; radius, spacing = 4radius, pattern = :hex, margin = radius, exclude = (), connect = exclude,
              base = nothing, bounds = (-1.0, 1.0), width = radius, frozen = nothing, rng = nothing, origin = nothing,
              jitter = 0.0, symmetry = nothing, nholes = nothing, min_distance = nothing,
              verbose = false) -> Vector{Float64}

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
  `(1 − jitter)(spacing − 2radius)`); the sequence of Julia's generators may change between versions of Julia, so a
  reference saved with `jitter` may change too.
- `pattern = :lattice`: square (cubic) lattice of step `spacing`, symmetric in the bounding box (with `jitter` too).
- `pattern = :cvt`: a centroidal Voronoi tessellation of the cells where a hole may go (Lloyd's relaxation): the
  centres spread evenly over any geometry, away from the excluded zones. `spacing` is the mean distance from a centre
  to its nearest neighbour (within 5 % where the number of holes allows it): the number of holes starts as that of
  `:hex` with the same `spacing` over that region (its measure over `√3/2 spacing²` in 2D, `spacing³/√2` in 3D), the
  mean distance is measured after the relaxation and the number corrected, at most twice (up to three relaxations;
  the closest run is kept). Slender parts get fewer holes than their measure suggests (a tube 1.5 `spacing` thick:
  half). `nholes` fixes the number of holes instead (images of a symmetry included; one relaxation). Neighbours are
  at least `min_distance` apart (default `2radius + (spacing − 2radius)/2`, half the solid of `:hex` between two
  holes; it must exceed `2radius`): after each Lloyd step the centres closer than that are pushed apart, and an orbit
  is dropped only where the minimum cannot be met. It starts from the centres of `:hex` (deterministic) or, with `rng`
  (an integer seed or an `AbstractRNG`), from random points at least `spacing` apart, thinned or completed by farthest
  points; at most 60 Lloyd steps per relaxation, until no centre moves more than 5 % of the mean cell size. Its
  centres are inside the region (near a centroid of a cell where a hole may go), not cut by the boundary as those of
  `:hex` may be. The sequence of Julia's generators may change between versions of Julia: give references without
  `rng`. `nholes` and `min_distance` are options of `:cvt` only.

`origin` (`:hex`, `:lattice`, and the start of `:cvt`) is a point of the lattice. A hole is kept when the centroid of
some cell is inside it (meshes that do not fill their bounding box), and holes cut by the boundary are kept (a
periodic pattern); to keep them inside, exclude the boundary tag. The holes keep `radius + margin` from the vertices
of the **frozen zones of `V`** (cells whose values are all Dirichlet values: read from `V`, nothing to give) and of
the tags in `exclude` (loads, supports), so that the solid is at least `margin` thick there. The distance is measured
to the vertices of those cells and tags: between two vertices `h` apart the distance to the entity can be up to
`R − √(R² − h²/4)` shorter (`R = radius + margin`; 3 % of `R` with `h = R/2`), which matters only on coarse meshes.

`symmetry`: one symmetry or a tuple of them, each a named tuple: `(point = p, normal = n)` a reflection in the plane
(line in 2D) through `p` normal to `n`; `(point = p, order = n)` a rotation of order `n` about `p` in 2D and
`(point = p, axis = a, order = n)` about the axis through `p` along `a` in 3D. The seed is built in a fundamental
region of the group they generate and replicated: the centres are exactly symmetric, a centre closer to a plane (or
an axis) than the distance between holes goes onto it, and every replica keeps the distance between holes and the
margins (an orbit that does not is dropped whole). The geometry, the frozen zones and the excluded tags must be
symmetric too: a warning says so when the mesh (to the largest cell size) or the keep-out points are not. With
`:hex` and `:lattice` the symmetry should also map the lattice onto itself (the planes of the default lattices through
the centre of the box, rotations of order 2 about it, and of order 4 for `:lattice`; `origin` moves the lattice): if
it does not, the replicas meet along seams where orbits are dropped and gaps are left, and a warning says how many;
`:cvt` has no such seams.
```julia
hole_seed(Vφ; radius = 0.05, spacing = 0.2, symmetry = (point = (1.0, 0.5), normal = (0.0, 1.0)))   # y = 0.5
hole_seed(Vt; radius = 0.1, pattern = :cvt, symmetry = (point = (0, 0, 0), axis = (0, 0, 1), order = 4))
```

The value at a point at distance `d` from the nearest centre goes from `lb` at `d = radius − width` to the interface
level `(lb + ub)/2` at `d = radius` and `ub` at `d = radius + width` (linear in `d`, clamped: a clipped distance for a
level set in `(-1, 1)`, a density in `(0, 1)` with `0.5` on the edge of the holes), and is then the minimum with
`base` (default `ub`; a number, a vector of free values or a function of the coordinates).

Checks: an `ArgumentError` when no hole fits, no value of the field falls inside a hole or every value is below the
level (all void); warnings when more than half of the values are below the level, when some holes
contain no value of the field (the mesh does not resolve them: refine or enlarge them), when the solid between two
holes is thinner than two cells, and when the solid (the cells whose mean value is at least the level, the frozen
cells with the values `frozen`, by default `ub`) does not join the tags of `connect` (by default `exclude`;
`nothing` skips the check) through faces. `verbose = true` prints the number of holes and the fraction of values in
them.

Void fraction of a periodic pattern with `spacing = 4radius`: `:hex` 23 % in 2D and 9 % in 3D, `:lattice` 20 % and
7 %; in 2D the best designs measured came from about four rows of holes across the smaller side of the domain
(`spacing ≈ L_min/4`). A pattern of cosines (Oka, Misawa, Yamada) is one line of the user,
`init = x -> clamp(-cos(2π x[1]/s) * cos(2π x[2]/s) + c, -1, 1)`, without the margins.

P0 and P1 Lagrangian spaces (and other scalar Lagrangian spaces: the values sit at the nodes), any mesh in 2D and 3D.
With periodic conditions the pattern is not periodic: the distance between holes is not kept across the periodic
sides. The cells of the model outside the triangulation of `V` (a design space on part of the model) count as solid
in the check of `connect`. The cost is linear in the cells and the values (a few microseconds per value; `:cvt`
that times its Lloyd steps, up to three relaxations). Sequential (not for distributed spaces yet).
"""
function hole_seed(V; radius::Real, spacing::Real = 4radius, pattern::Symbol = :hex, margin::Real = radius,
                   exclude = (), connect = exclude, base = nothing, bounds = (-1.0, 1.0), width::Real = radius,
                   frozen = nothing, rng = nothing, origin = nothing, jitter::Real = 0.0, symmetry = nothing,
                   nholes = nothing, min_distance = nothing, verbose::Bool = false)
  lb, ub, r = _seed_args(V, radius, spacing, pattern, margin, width, jitter, rng, bounds, nholes, min_distance)
  radius, spacing, margin, width, jitter = Float64.((radius, spacing, margin, width, jitter))
  G = _seed_geometry(V)
  _check_tags(G.model, _seed_tags(exclude), "exclude")
  co = _seed_tags(connect)
  _check_tags(G.model, co, "connect")
  hmean = G.hmean
  ligament = pattern === :cvt ? (min_distance === nothing ? (spacing - 2radius) / 2 : min_distance - 2radius) :
             (1 - jitter) * (spacing - 2radius)
  ligament >= 2hmean ||
    @warn "hole_seed: the solid between two holes ($(round(ligament, sigdigits = 3))) is thinner than two cells (mean cell size $(round(hmean, sigdigits = 3)))"
  centers = _seed_centers(G, radius, spacing, pattern, margin, exclude, r, origin, jitter, symmetry, nholes, min_distance)
  isempty(centers) &&
    throw(ArgumentError("hole_seed: no hole fits (a centroid of a cell inside each hole, radius + margin = $(radius + margin) " *
                        "from the frozen zones and the excluded tags)"))
  n = num_free_dofs(V)
  bv = base === nothing ? nothing : base isa Number ? nothing : base isa AbstractVector ? base :
       get_free_dof_values(interpolate(base, V))
  bv === nothing || length(bv) == n ||
    throw(DimensionMismatch("hole_seed: base has $(length(bv)) values, the space has $n free values"))
  b0 = base isa Number ? Float64(base) : ub
  φ, nin, empty = _seed_values(V, G, G.dim, centers, radius, width, lb, ub, bv, b0)
  level = (lb + ub) / 2
  nin > 0 || throw(ArgumentError("hole_seed: no value of the field falls inside a hole (radius = $radius, mean cell " *
                                 "size $(round(hmean, sigdigits = 3))): refine the mesh or enlarge the holes"))
  nvoid = count(<(level), φ)
  nvoid < n || throw(ArgumentError("hole_seed: every value of the field is below the level (all void): the holes " *
                                   "cover the domain (radius = $radius, spacing = $spacing, lengths in the units of " *
                                   "the mesh): reduce radius or enlarge spacing"))
  nvoid <= n / 2 ||
    @warn "hole_seed: $(round(100nvoid / n, digits = 1)) % of the values are below the level (more void than solid): " *
          "check radius and spacing (radius = $radius, spacing = $spacing, lengths in the units of the mesh)"
  empty > 0 &&
    @warn "hole_seed: $empty of $(length(centers)) holes contain no value of the field (radius = $radius, mean cell size $(round(hmean, sigdigits = 3))): refine the mesh or enlarge the holes"
  connected = true
  if !isempty(co)
    connected, why = _solid_connects(V, G, φ, level, frozen === nothing ? ub : frozen, co)
    connected || @warn "hole_seed: the solid of the seed does not join the tags $(join(co, ", ")): $why"
  end
  verbose && @info "hole_seed: $(length(centers)) holes ($pattern), $(round(100nvoid / n, digits = 1)) % of " *
                   "the values below the level" * (isempty(co) ? "" : connected ? ", connected" : ", not connected")
  φ
end

# The values (a function barrier: D from the type of the centres). The coordinates of the free values by interpolation
# (P0, P1, any numbering); the number of values inside the holes and the number of holes without any.
function _seed_values(V, G, ::Val{D}, centers, radius, width, lb, ub, bv, b0) where {D}
  n = num_free_dofs(V)
  xs = ntuple(k -> collect(Float64, get_free_dof_values(interpolate(x -> x[k], V)))::Vector{Float64}, Val(D))
  B = _Buckets(G.lo, G.hi, radius + width, length(centers))
  for c in centers; _push!(B, c); end
  φ = Vector{Float64}(undef, n)
  @inbounds for i in 1:n
    x = ntuple(k -> xs[k][i], Val(D))
    d2 = _nearest2(B, x)
    s = d2 == Inf ? 1.0 : clamp((sqrt(d2) - radius) / width, -1.0, 1.0)
    v = lb + (ub - lb) * (s + 1) / 2
    φ[i] = clamp(min(bv === nothing ? b0 : Float64(bv[i]), v), lb, ub)
  end
  # holes without any value of the field inside
  E = _Buckets(G.lo, G.hi, radius, n)
  nin = 0
  @inbounds for i in 1:n
    x = ntuple(k -> xs[k][i], Val(D))
    _nearest2(B, x) < radius^2 || continue
    nin += 1
    _push!(E, x)
  end
  empty = count(c -> _nearest2(E, c) >= radius^2, centers)
  φ, nin, empty
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
  solid = trues(ncm)                         # cells of the model outside the triangulation of V: solid (not design)
  ci = Gridap.Arrays.array_cache(G.ids)
  for c in 1:length(G.ids)
    cell = Gridap.Arrays.getindex!(ci, G.ids, c)
    s = 0.0
    for d in cell
      s += d > 0 ? φ[d] : dv === nothing ? Float64(frozen) : dv[-d]
    end
    solid[t2m[c]] = s / length(cell) >= level
  end
  c2f = Gridap.Geometry.get_faces(topo, D, D - 1)
  f2c = Gridap.Geometry.get_faces(topo, D - 1, D)
  cc2f, cf2c = Gridap.Arrays.array_cache(c2f), Gridap.Arrays.array_cache(f2c)
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
      for f in Gridap.Arrays.getindex!(cc2f, c2f, c), c2 in Gridap.Arrays.getindex!(cf2c, f2c, f)
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
