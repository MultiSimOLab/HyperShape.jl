# Exact fraction of a cell where a linear (or subdivided) nodal field is ≥ 0, and its derivative with respect to the
# nodal values (the kernels of Ersatz()). Convention: φ ≥ 0 is solid. Closed forms without cancellation (every term
# of a sum has the same sign); tuples in, tuples out, no allocations.

# one vertex a > 0, the others ≤ 0:  f = a^D / Π (a − b_j)
@inline function _corner(a, b, c)                 # triangle → f, (∂a, ∂b, ∂c)
  d1, d2 = a - b, a - c
  f = a * a / (d1 * d2)
  fa = a * (2b * c - a * (b + c)) / (d1 * d1 * d2 * d2)
  f, (fa, f / d1, f / d2)
end
@inline function _corner(a, b, c, d)              # tetrahedron
  d1, d2, d3 = a - b, a - c, a - d
  p = d1 * d2 * d3
  f = a^3 / p
  s1, s2, s3 = b + c + d, b * c + b * d + c * d, b * c * d
  fa = a * a * (-a * a * s1 + 2a * s2 - 3s3) / (p * p)
  f, (fa, f / d1, f / d2, f / d3)
end
# tetrahedron, a, b > 0 ≥ c, d:  f = [a²b² − ab(a+b)(c+d) + cd(a²+ab+b²)] / ((a−c)(a−d)(b−c)(b−d))
@inline function _wedge(a, b, c, d)
  ac, ad, bc, bd = a - c, a - d, b - c, b - d
  den = ac * ad * bc * bd
  N = a * a * b * b - a * b * (a + b) * (c + d) + c * d * (a * a + a * b + b * b)
  f = N / den
  Na = 2a * b * b - (2a * b + b * b) * (c + d) + c * d * (2a + b)
  Nb = 2a * a * b - (a * a + 2a * b) * (c + d) + c * d * (a + 2b)
  Nc = -a * b * (a + b) + d * (a * a + a * b + b * b)
  Nd = -a * b * (a + b) + c * (a * a + a * b + b * b)
  f, (Na / den - f * (1 / ac + 1 / ad), Nb / den - f * (1 / bc + 1 / bd),
      Nc / den + f * (1 / ac + 1 / bc), Nd / den + f * (1 / ad + 1 / bd))
end

"""
    cell_fraction(φ::NTuple{3,Float64}) -> (f, ∂f/∂φ)
    cell_fraction(φ::NTuple{4,Float64}) -> (f, ∂f/∂φ)

Fraction of a triangle (3 values) or a tetrahedron (4 values) where the linear interpolant of the vertex values `φ`
is ≥ 0, and its derivative with respect to them. A cell without interface has derivative zero; `φ ≡ 0` gives 1.
"""
function cell_fraction(φ::NTuple{3,Float64})
  p1, p2, p3 = φ[1] >= 0, φ[2] >= 0, φ[3] >= 0
  n = p1 + p2 + p3
  n == 3 && return 1.0, (0.0, 0.0, 0.0)
  n == 0 && return 0.0, (0.0, 0.0, 0.0)
  s = n == 1 ? 1.0 : -1.0                         # two vertices ≥ 0: the complement, with −φ
  i = n == 1 ? (p1 ? 1 : p2 ? 2 : 3) : (!p1 ? 1 : !p2 ? 2 : 3)
  j, k = i == 1 ? (2, 3) : i == 2 ? (1, 3) : (1, 2)
  a = s * φ[i]
  a == 0 && return (n == 1 ? 0.0 : 1.0), (0.0, 0.0, 0.0)   # the interface touches one vertex only
  f, g = _corner(a, s * φ[j], s * φ[k])
  out = (i == 1 ? g[1] : j == 1 ? g[2] : g[3], i == 2 ? g[1] : j == 2 ? g[2] : g[3], i == 3 ? g[1] : j == 3 ? g[2] : g[3])
  (n == 1 ? f : 1 - f), out                       # d(1 − f(−φ))/dφ = +g
end

@inline _place(g::NTuple{4,Float64}, perm::NTuple{4,Int}) =
  Base.setindex(Base.setindex(Base.setindex(Base.setindex((0.0, 0.0, 0.0, 0.0), g[1], perm[1]), g[2], perm[2]), g[3], perm[3]), g[4], perm[4])

function cell_fraction(φ::NTuple{4,Float64})
  z = (0.0, 0.0, 0.0, 0.0)
  perm = (0, 0, 0, 0)
  k = 0                                           # vertices with φ ≥ 0 first
  for m in 1:4
    φ[m] >= 0 && (k += 1; perm = Base.setindex(perm, m, k))
  end
  n = k
  n == 4 && return 1.0, z
  n == 0 && return 0.0, z
  for m in 1:4
    φ[m] >= 0 || (k += 1; perm = Base.setindex(perm, m, k))
  end
  a, b, c, d = φ[perm[1]], φ[perm[2]], φ[perm[3]], φ[perm[4]]
  if n == 1
    a == 0 && return 0.0, z                       # the interface touches one vertex only
    f, g = _corner(a, b, c, d)
    return f, _place(g, perm)
  elseif n == 2
    (a == 0 && b == 0) && return 0.0, z           # the interface is an edge
    f, g = _wedge(a, b, c, d)
    return f, _place(g, perm)
  else
    f, g = _corner(-d, -a, -b, -c)                # the complement, with −φ: d(1 − f(−φ))/dφ = +g
    return 1 - f, _place((g[2], g[3], g[4], g[1]), perm)
  end
end

# Quadrilateral (vertex order of Gridap: (0,0), (1,0), (0,1), (1,1)): 4 triangles around the centre, whose value is
# the mean of the 4 nodes (the bilinear value there). w: relative areas of the 4 triangles (1/4 on a parallelogram).
const QUAD_EDGES = ((1, 2), (2, 4), (4, 3), (3, 1))
function quad_fraction(φ::NTuple{4,Float64}, w::NTuple{4,Float64})
  c = 0.25 * (φ[1] + φ[2] + φ[3] + φ[4])
  f1, g1 = cell_fraction((φ[1], φ[2], c))
  f2, g2 = cell_fraction((φ[2], φ[4], c))
  f3, g3 = cell_fraction((φ[4], φ[3], c))
  f4, g4 = cell_fraction((φ[3], φ[1], c))
  gc = 0.25 * (w[1] * g1[3] + w[2] * g2[3] + w[3] * g3[3] + w[4] * g4[3])
  w[1] * f1 + w[2] * f2 + w[3] * f3 + w[4] * f4,
  (w[1] * g1[1] + w[4] * g4[2] + gc, w[1] * g1[2] + w[2] * g2[1] + gc, w[3] * g3[2] + w[4] * g4[1] + gc,
   w[2] * g2[2] + w[3] * g3[1] + gc)
end

# Hexahedron (vertex order of Gridap, x fastest): 24 tetrahedra (cell centre, face centre, one edge of the face).
# w: relative volumes of the 24 tetrahedra, in the order of the loops below (1/24 on a parallelepiped).
const HEX_FACES = ((1, 2, 4, 3), (5, 6, 8, 7), (1, 2, 6, 5), (3, 4, 8, 7), (1, 3, 7, 5), (2, 4, 8, 6))  # cyclic
function hex_fraction(φ::NTuple{8,Float64}, w::NTuple{24,Float64})
  z8 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  all(>=(0), φ) && return 1.0, z8                 # no interface: exact, and 24 times cheaper
  all(<(0), φ) && return 0.0, z8
  c = (φ[1] + φ[2] + φ[3] + φ[4] + φ[5] + φ[6] + φ[7] + φ[8]) / 8
  f = 0.0
  gc = 0.0
  g = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  t = 0
  for F in HEX_FACES
    fc = 0.25 * (φ[F[1]] + φ[F[2]] + φ[F[3]] + φ[F[4]])
    gf = 0.0
    for e in 1:4
      t += 1
      i, j = F[e], F[e == 4 ? 1 : e + 1]
      ft, gt = cell_fraction((φ[i], φ[j], fc, c))
      wt = w[t]
      f += wt * ft
      g = Base.setindex(g, g[i] + wt * gt[1], i)
      g = Base.setindex(g, g[j] + wt * gt[2], j)
      gf += wt * gt[3]
      gc += wt * gt[4]
    end
    for e in 1:4
      g = Base.setindex(g, g[F[e]] + 0.25 * gf, F[e])
    end
  end
  f, g .+ gc / 8
end

# relative areas / volumes of the subdivision of a cell from its vertex coordinates (tuples of points)
_tri_area(a, b, c) = abs((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])) / 2
_tet_vol(a, b, c, d) = abs(_det3(b - a, c - a, d - a)) / 6
_det3(u, v, w) = u[1] * (v[2] * w[3] - v[3] * w[2]) - u[2] * (v[1] * w[3] - v[3] * w[1]) + u[3] * (v[1] * w[2] - v[2] * w[1])

function _quad_weights(X)
  c = (X[1] + X[2] + X[3] + X[4]) / 4
  a = ntuple(e -> _tri_area(X[QUAD_EDGES[e][1]], X[QUAD_EDGES[e][2]], c), 4)
  s = sum(a)
  s > 0 || throw(ArgumentError("Ersatz: a quadrilateral has zero area"))
  a ./ s
end

function _hex_weights(X)
  c = sum(X) / 8
  v = zeros(24)
  t = 0
  for F in HEX_FACES
    fc = (X[F[1]] + X[F[2]] + X[F[3]] + X[F[4]]) / 4
    for e in 1:4
      t += 1
      v[t] = _tet_vol(X[F[e]], X[F[e == 4 ? 1 : e + 1]], fc, c)
    end
  end
  s = sum(v)
  s > 0 || throw(ArgumentError("Ersatz: a hexahedron has zero volume"))
  NTuple{24,Float64}(v ./ s)
end
