"""
    FieldVariable(V; bounds, init)

Block of [`OptVariables`](@ref) holding the free values of a scalar design field in the space `V` (cell-wise P0 or
nodal P1 Lagrangian, without Dirichlet conditions; the bounds act on the free values, which bound the field only for
these two spaces). `bounds = (lb, ub)` with `lb < ub`; `init` is a number or a vector of the length of
the free values, inside the bounds. Both are mandatory.
"""
struct FieldVariable{S}
  space::S
  lb::Float64
  ub::Float64
  init::Vector{Float64}
end

function FieldVariable(V; bounds, init)
  num_dirichlet_dofs(V) == 0 ||
    throw(ArgumentError("FieldVariable: the design space must not have Dirichlet conditions"))
  lb, ub = Float64(bounds[1]), Float64(bounds[2])
  lb < ub || throw(ArgumentError("FieldVariable: bounds must satisfy lb < ub, got $bounds"))
  n = num_free_dofs(V)
  x0 = init isa Number ? fill(Float64(init), n) : Vector{Float64}(init)
  length(x0) == n || throw(DimensionMismatch("FieldVariable: init has $(length(x0)) values, the space has $n"))
  all(v -> lb <= v <= ub, x0) || throw(ArgumentError("FieldVariable: init outside the bounds $bounds"))
  FieldVariable(V, lb, ub, x0)
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

version(z::OptVariables) = z.version
Base.length(z::OptVariables) = length(z.values)
get_values(z::OptVariables) = z.values
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
Base.show(io::IO, v::FieldVariable) = print(io, "FieldVariable(", length(v.init), " values, bounds (", v.lb, ", ", v.ub, "))")
