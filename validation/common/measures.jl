# Measures of a design for the validation cases and the fixed report table (plain Gridap and the public API of
# HyperShape; not a library). A design is read through its cell means χ on a Cartesian mesh of n = (nx, ny[, nz])
# cells of size hcell (Gridap's order: x fastest); "solid" means χ ≥ ½.

# cell means of the physical density
cell_means(θ, dΩ, vol) = collect(Gridap.get_array(∫(θ.ρ)dΩ)) ./ vol

_solid(c::Bool) = c
_solid(c::Real) = c >= 0.5

"""
    binary_evaluator(form, l, U, V, V0; Emin)

The objective of a design read as 0/1 per cell (`J_bin`): a function `χ -> J` that solves `form(E, u, v) = l(v)` on
the spaces `U`, `V` with the cell-wise modulus `E = Emin + (1 − Emin) χ_bin` (`χ_bin = χ ≥ ½`, or a `Bool` per cell;
`V0` the P0 space of the cells) and returns `l(u)` (the compliance). Plain Gridap: it does not use the state of
HyperShape, so it is the common evaluation of any method on the same mesh and loads.
"""
binary_evaluator(form, l, U, V, V0; Emin) = χ -> begin
  Eh = FEFunction(V0, [Emin + (_solid(c) ? 1 - Emin : 0.0) for c in vec(χ)])
  uh = solve(AffineFEOperator((u, v) -> form(Eh, u, v), l, U, V))
  sum(l(uh))
end

"""
    density_evaluator(form, l, U, V, V0; Emin, p = 3)

The objective of a design given by one density per cell (a design of another code, such as `xPhys` of top88): a
function `ρ -> J` that solves `form(E, u, v) = l(v)` with the cell-wise SIMP modulus `E = Emin + (1 − Emin) ρᵖ` and
returns `l(u)`. Plain Gridap, like `binary_evaluator`.
"""
density_evaluator(form, l, U, V, V0; Emin, p = 3) = ρ -> begin
  Eh = FEFunction(V0, [Emin + (1 - Emin) * c^p for c in vec(ρ)])
  uh = solve(AffineFEOperator((u, v) -> form(Eh, u, v), l, U, V))
  sum(l(uh))
end

"fraction of cells with χ ≥ ½ (uniform mesh)"
binary_volume(χ) = count(_solid, χ) / length(χ)

"Sigmund's measure of grey of the cell means, `mean(4χ(1 − χ))` (0 for a 0/1 design, 1 for χ = ½ everywhere)"
grey(χ) = sum(c -> 4c * (1 - c), χ) / length(χ)

"""
    perimeter(χ, n, hcell)

Perimeter (area in 3D) of the 0/1 design in domain units: the interior faces between a solid and a void cell times
their measure. A staircase measure: comparable between methods on the same mesh; under refinement it tends to the ℓ¹
length of the boundary, not to the Euclidean one.
"""
function perimeter(χ, n::Tuple, hcell::Tuple)
  S = reshape(map(_solid, vec(χ)), n)
  P = 0.0
  for d in 1:length(n)
    face = prod(hcell) / hcell[d]
    a = selectdim(S, d, 1:n[d]-1)
    b = selectdim(S, d, 2:n[d])
    P += count(a .!= b) * face
  end
  P
end

"""
    interior_holes(χ, nx, ny; min_cells = 1)
    interior_holes(χ::AbstractArray; min_cells = 1)

Interior holes of a design given by its cell means (or a `Bool` array, `true` = solid) on a Cartesian mesh: the
connected components (by faces) of the cells below ½ that do not touch the boundary and have at least `min_cells`
cells. Works in 2D and 3D (in 3D it counts closed cavities).
"""
interior_holes(χ, nx, ny; min_cells = 1) = interior_holes(reshape(χ, nx, ny); min_cells)
function interior_holes(χ::AbstractArray; min_cells = 1)
  void = map(c -> !_solid(c), χ)
  seen = falses(size(void))
  nb = [CartesianIndex(ntuple(k -> k == d ? s : 0, ndims(void))) for d in 1:ndims(void) for s in (1, -1)]
  n = 0
  for I in CartesianIndices(void)
    (void[I] && !seen[I]) || continue
    stack = [I]
    seen[I] = true
    inner = true
    size_ = 0
    while !isempty(stack)
      J = pop!(stack)
      size_ += 1
      for d in nb
        K = J + d
        checkbounds(Bool, void, K) || (inner = false; continue)
        (void[K] && !seen[K]) || continue
        seen[K] = true
        push!(stack, K)
      end
    end
    inner && size_ >= min_cells && (n += 1)
  end
  n
end
