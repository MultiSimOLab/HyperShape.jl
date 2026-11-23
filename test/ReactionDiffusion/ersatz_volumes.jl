# Ersatz(): volumes of known regions, on Cartesian meshes and on unstructured meshes of Gmsh (test/meshes, Gridap
# .json written from .msh files: m<dim>_<s: simplices | q: quadrilaterals>_<1/h>)
using Test
using Gridap
using HyperShape

const MESHES = joinpath(@__DIR__, "..", "meshes")

function ersatz_volume(model, f)
  V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1)
  z = OptVariables(φ = FieldVariable(V; bounds = (-10.0, 10.0), init = f))
  θ = PhysicalFields(DesignFields(z; φ = Identity()); ρ = :φ => Ersatz())
  sum(∫(θ.ρ)design_measures(θ; degree = 1).domain)
end

cartesian(dim, n, simplex) = (m = CartesianDiscreteModel(Tuple(repeat([0, 1], dim)), Tuple(fill(n, dim))); simplex ? simplexify(m) : m)
gmsh_mesh(name) = DiscreteModelFromFile(joinpath(MESHES, name * ".json"))

# the volume of {c⋅x + d ≥ 0} in the unit cube (c > 0), by inclusion–exclusion
function halfspace_volume(c, d)
  n = length(c)
  t = -d
  s = 0.0
  for mask in 0:(2^n - 1)
    S = [i for i in 1:n if (mask >> (i - 1)) & 1 == 1]
    s += (-1)^length(S) * max(0.0, t - sum(c[S]; init = 0.0))^n
  end
  1 - s / (factorial(n) * prod(c))
end

@testset "Ersatz(): volumes" begin
  @testset "inclined half space, exact ($dim D, $name)" for (dim, name, mk) in (
      (2, "Cartesian triangles", () -> cartesian(2, 8, true)), (2, "Cartesian quadrilaterals", () -> cartesian(2, 8, false)),
      (2, "Gmsh triangles", () -> gmsh_mesh("m2_s_8")), (2, "Gmsh quadrilaterals", () -> gmsh_mesh("m2_q_10")),
      (3, "Cartesian tetrahedra", () -> cartesian(3, 4, true)), (3, "Cartesian hexahedra", () -> cartesian(3, 4, false)),
      (3, "Gmsh tetrahedra", () -> gmsh_mesh("m3_s_4")))
    c = (0.31, 0.27, 0.22)[1:dim]
    d = -0.5 * sum(c) + 0.05
    exact = halfspace_volume(c, d)
    v = ersatz_volume(mk(), x -> sum(c[i] * x[i] for i in 1:dim) + d)
    @test abs(v - exact) <= 1e-13 * exact
  end

  # a disk and a sphere of radius 0.3: second order (measured on 2026-11-22: 2.00 Cartesian quadrilaterals 8→16,
  # 2.02 hexahedra 4→8, 2.00 Gmsh triangles 8→16, 1.96 Gmsh tetrahedra 4→8)
  R = 0.3
  @testset "disk and sphere, order ≥ 1.9 ($dim D, $name)" for (dim, name, coarse, fine) in (
      (2, "Cartesian quadrilaterals", () -> cartesian(2, 8, false), () -> cartesian(2, 16, false)),
      (3, "Cartesian hexahedra", () -> cartesian(3, 4, false), () -> cartesian(3, 8, false)),
      (2, "Gmsh triangles", () -> gmsh_mesh("m2_s_8"), () -> gmsh_mesh("m2_s_16")),
      (3, "Gmsh tetrahedra", () -> gmsh_mesh("m3_s_4"), () -> gmsh_mesh("m3_s_8")))
    exact = dim == 2 ? π * R^2 : 4 / 3 * π * R^3
    f(x) = R - sqrt(sum((x[i] - 0.5)^2 for i in 1:dim))
    e1 = abs(ersatz_volume(coarse(), f) - exact) / exact
    e2 = abs(ersatz_volume(fine(), f) - exact) / exact
    @test log2(e1 / e2) >= 1.9
  end
end
