"""
    FieldVariable(V; bounds, init, frozen = nothing)

Block of [`OptVariables`](@ref) holding the free values of a scalar design field in the space `V` (cell-wise P0 or
nodal P1 Lagrangian; the bounds act on the free values, which bound the field only for these two spaces).
`bounds = (lb, ub)` with `lb < ub`; `init` is a number, a vector of the length of the free values or a function of the
coordinates (interpolated on `V`), inside the bounds. Both are mandatory.

**Frozen zones** (solid or void that the optimisation cannot change, such as the material under a load or at a
support): build `V` with `dirichlet_tags` on the zones and give their values in `frozen` (a number, one number per tag
in the order of `dirichlet_tags`, or a function of the coordinates), inside the bounds:

```julia
add_tag_from_vertex_filter!(labels, "solid", model, x -> x[2] ≥ ny - 2 && x[1] ≤ 2)
Vρ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity = :L2, dirichlet_tags = "solid")
z, x, θ = single_field_design(:ρ, Vρ; bounds = (0.0, 1.0), init = 0.5, frozen = 1.0, map = …)
```

The values of the zones are not variables (`z` holds the free ones only), the design field `x` has the frozen values
there, and every physical field built from it takes, on the cells of the zones, the image of the frozen value by the
pointwise maps of its chain (projection and pointwise maps; filters and cell means leave it as it is): the zone is
real material whatever the filter (see [`PhysicalFields`](@ref)). A zone is made of whole cells: every Dirichlet
value must belong to a cell all of whose values are frozen (a tag of vertices made with `add_tag_from_vertex_filter!`
gives that for P0 and P1; a line of nodes on a boundary does not, and is an error: turn a boundary tag into the cells
next to it with [`tag_adjacent_cells!`](@ref); in P1 a cell outside the tags whose nodes are all frozen, such as a hole
of one cell inside a zone, belongs to the zone too), and its value is one number per
cell (zones with different values must not touch; a function of the coordinates must be constant on each frozen
cell). Spaces with periodic conditions are fine; spaces with other linear constraints (symmetries) are not supported
yet. With a frozen zone the space of the block is the `TrialFESpace` with the frozen values (what `metric_weights`
of a metric of the user receives).
"""
struct FieldVariable{S,Z}
  space::S
  lb::Float64
  ub::Float64
  init::Vector{Float64}
  zone::Z               # nothing, or the frozen zone (FrozenZone)
end

# A frozen zone of a design space: the cells whose design values are all Dirichlet values
struct FrozenZone{C,F,G}
  χ::C          # P0 indicator of the frozen cells
  free::F       # 1 - χ
  value::G      # P0 frozen value of each frozen cell (0 elsewhere)
  ncells::Int
  ndofs::Int
end

function FieldVariable(V; bounds, init, frozen = nothing)
  lb, ub = Float64(bounds[1]), Float64(bounds[2])
  lb < ub || throw(ArgumentError("FieldVariable: bounds must satisfy lb < ub, got $bounds"))
  V isa Gridap.FESpaces.FESpaceWithLinearConstraints &&
    throw(ArgumentError("FieldVariable: design spaces with linear constraints (symmetries, hanging nodes) are not supported yet"))
  nd = num_dirichlet_dofs(V)
  if nd == 0
    frozen === nothing ||
      throw(ArgumentError("FieldVariable: frozen is given but the design space has no Dirichlet tags (build it with dirichlet_tags = …)"))
    U, zone = V, nothing
  else
    V isa TrialFESpace &&
      throw(ArgumentError("FieldVariable: give the test space (with dirichlet_tags) and the values in frozen"))
    frozen === nothing && throw(ArgumentError("FieldVariable: the design space has Dirichlet tags (frozen zones); " *
                                              "give their values: frozen = 1.0 (or one per tag, or a function of x)"))
    U = try
      TrialFESpace(V, frozen)
    catch e
      (e isa AssertionError && occursin("dirichlet tags", e.msg)) || rethrow()
      throw(ArgumentError("FieldVariable: frozen must be a number, a function of x or one value per Dirichlet tag " *
                          "(in the order of dirichlet_tags)"))
    end
    dv = get_dirichlet_dof_values(U)
    all(v -> lb <= v <= ub, dv) || throw(ArgumentError("FieldVariable: frozen values outside the bounds $bounds"))
    zone = _frozen_zone(U, dv)
  end
  n = num_free_dofs(U)
  # a function (or a callable object) of the coordinates
  init isa Union{Number,AbstractVector} || (init = get_free_dof_values(interpolate(init, U)))
  x0 = init isa Number ? fill(Float64(init), n) : Vector{Float64}(init)
  length(x0) == n || throw(DimensionMismatch("FieldVariable: init has $(length(x0)) values, the space has $n"))
  all(v -> lb <= v <= ub, x0) ||
    throw(ArgumentError("FieldVariable: init outside the bounds $bounds (its values go from $(minimum(x0)) to " *
                        "$(maximum(x0)); clamp it)"))
  FieldVariable(U, lb, ub, x0, zone)
end

function _frozen_zone(U, dv)
  Ω = get_triangulation(U)
  V0 = FESpace(Ω, ReferenceFE(lagrangian, Float64, 0); conformity = :L2)
  ids, ids0 = get_cell_dof_ids(U), get_cell_dof_ids(V0)
  χv, gv = zeros(num_free_dofs(V0)), zeros(num_free_dofs(V0))
  covered = falses(length(dv))
  for c in 1:length(ids)
    cell = ids[c]
    all(<(0), cell) || continue
    k = ids0[c][1]
    χv[k] = 1.0
    v1 = dv[-first(cell)]
    for d in cell
      covered[-d] = true
      abs(dv[-d] - v1) <= 1e-12 * max(1.0, abs(v1)) || throw(ArgumentError("FieldVariable: a frozen cell has " *
        "different frozen values ($v1 and $(dv[-d])): the value of a zone is one number per cell (zones with different " *
        "values must not touch; a function of x must be constant on each cell)"))
    end
    gv[k] = v1
  end
  all(covered) || throw(ArgumentError("FieldVariable: $(count(!, covered)) frozen values do not belong to a frozen cell: " *
    "a frozen zone must be made of whole cells (a line of nodes on a boundary is not a zone; tag the cells next to it " *
    "with tag_adjacent_cells!(labels, newtag, model, tag))"))
  FrozenZone(FEFunction(V0, χv), FEFunction(V0, 1 .- χv), FEFunction(V0, gv), Int(sum(χv)), length(dv))
end

"""
    OptVariables(; name = FieldVariable(...), ...)

What the optimiser sees: one vector `z` made of named blocks, with bounds. Read it with `get_values(z)` and
`get_bounds(z)`. After writing into `get_values(z)` in place, call `changed!(z)` so that the design and physical
fields are updated; `set_values!(z, v)` does both.
"""
mutable struct OptVariables{N,B<:NTuple{N,FieldVariable}}
  names::NTuple{N,Symbol}
  blocks::B
  ranges::NTuple{N,UnitRange{Int}}
  values::Vector{Float64}
  lb::Vector{Float64}
  ub::Vector{Float64}
  version::Int
end

function OptVariables(; kwargs...)
  isempty(kwargs) && throw(ArgumentError("OptVariables needs at least one block"))
  names = Tuple(keys(kwargs))
  blocks = Tuple(values(kwargs))
  for (k, b) in zip(names, blocks)
    b isa FieldVariable || throw(ArgumentError("OptVariables: block $k must be a FieldVariable, got $(typeof(b))"))
  end
  ranges = UnitRange{Int}[]
  i = 0
  for b in blocks
    n = length(b.init)
    push!(ranges, (i + 1):(i + n))
    i += n
  end
  vals = reduce(vcat, (b.init for b in blocks))
  lb = reduce(vcat, (fill(b.lb, length(b.init)) for b in blocks))
  ub = reduce(vcat, (fill(b.ub, length(b.init)) for b in blocks))
  OptVariables(names, blocks, Tuple(ranges), vals, lb, ub, 0)
end

state_version(z::OptVariables) = z.version
Base.length(z::OptVariables) = length(z.values)
"""
    get_values(z::OptVariables) -> Vector{Float64}

The design vector, all blocks in order (the same object, not a copy). After writing into it in place call
`changed!(z)`; or use [`set_values!`](@ref).
"""
get_values(z::OptVariables) = z.values

"""
    get_bounds(z::OptVariables) -> (lb, ub)

Lower and upper bounds of every entry of the design vector.
"""
get_bounds(z::OptVariables) = (z.lb, z.ub)
block_range(z::OptVariables, name::Symbol) = z.ranges[_index(z.names, name)]

"""
    changed!(z::OptVariables)

Marks the values of `z` as changed (after writing into `get_values(z)` in place). Throws if a value is outside the
bounds or not finite.
"""
function changed!(z::OptVariables)
  _check_values(z)
  z.version += 1
  z
end

function _check_values(z::OptVariables)
  v, lb, ub = z.values, z.lb, z.ub
  ok = true
  @inbounds for i in eachindex(v, lb, ub)
    ok &= lb[i] <= v[i] <= ub[i]   # false for NaN
  end
  ok || throw(ArgumentError("OptVariables: values outside the bounds or not finite"))
  nothing
end

"""
    set_values!(z::OptVariables, v)

Copies `v` into `z` and marks it as changed (values must be inside the bounds).
"""
function set_values!(z::OptVariables, v)
  length(v) == length(z.values) || throw(DimensionMismatch("set_values!: $(length(v)) values, z has $(length(z.values))"))
  copyto!(z.values, v)
  changed!(z)
end

function _index(names, name)
  i = findfirst(==(name), names)
  i === nothing && throw(ArgumentError("no block or field named $name; available: $(join(names, ", "))"))
  i
end

Base.show(io::IO, z::OptVariables) =
  print(io, "OptVariables(", join(("$n ($(length(r)))" for (n, r) in zip(z.names, z.ranges)), ", "), ")")
Base.show(io::IO, v::FieldVariable) = print(io, "FieldVariable(", length(v.init), " values, bounds (", v.lb, ", ", v.ub, ")",
                                             v.zone === nothing ? "" : ", $(v.zone.ncells) frozen cells", ")")
