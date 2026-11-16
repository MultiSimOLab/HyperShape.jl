# Fast field writer: the mesh once (continuous, with the node numbering of the grid), the values straight from the
# degrees of freedom, optionally in another task. Formats: VTU (one file per step, with a .pvd) and VTKHDF (one file
# with every step; needs `using HDF5`, see ext/HyperShapeHDF5Ext.jl).

struct NotFast <: Exception
  msg::String
end

# ---------------------------------------------------------------------------------------------------------------
# Mesh, written once

struct WriterMesh{C<:AbstractVector{<:WriteVTK.MeshCell}}
  points::Matrix{Float64}          # 3 × np (always Float64)
  cn::Matrix{Int32}                # nv × nc: writer node of each vertex of each cell, in Gridap's vertex order
  conn::Vector{Int}                # 0-based connectivity, VTK order
  nv::Int
  ctype::UInt8                     # VTK cell type
  cells::C                         # concrete element type: WriteVTK is slow with an abstract one
  verts::Any                       # reference coordinates of the vertices (Gridap order)
end

npoints(m::WriterMesh) = size(m.points, 2)
ncells(m::WriterMesh) = size(m.cn, 2)

const _VTK = WriteVTK.VTKCellTypes

# Gridap vertex order → VTK order (n-cubes are numbered lexicographically in Gridap)
function _vtk_order(p)
  D = num_dims(p)
  if is_n_cube(p)
    D == 1 && return [1, 2], _VTK.VTK_LINE
    D == 2 && return [1, 2, 4, 3], _VTK.VTK_QUAD
    D == 3 && return [1, 2, 4, 3, 5, 6, 8, 7], _VTK.VTK_HEXAHEDRON
  else
    D == 2 && return [1, 2, 3], _VTK.VTK_TRIANGLE
    D == 3 && return [1, 2, 3, 4], _VTK.VTK_TETRA
  end
  throw(NotFast("cell type not supported"))
end

function WriterMesh(Ω::Triangulation)
  grid = get_grid(Ω)
  reffes = get_reffes(grid)
  length(reffes) == 1 || throw(ArgumentError("FieldWriter: one cell type per triangulation (use createvtk for mixed meshes)"))
  num_cell_dims(Ω) == num_point_dims(Ω) ||
    throw(ArgumentError("FieldWriter: the cells of Ω must have the dimension of the space (use createvtk on boundaries)"))
  p = get_polytope(only(reffes))
  nv = num_vertices(p)
  perm, wt = _vtk_order(p)
  X = get_node_coordinates(grid)
  cnl = get_cell_node_ids(grid)
  nc = length(cnl)
  D = length(eltype(X))
  g2w = zeros(Int32, length(X))
  cn = zeros(Int32, nv, nc)
  np = 0
  cache = array_cache(cnl)
  for c in 1:nc
    ids = getindex!(cache, cnl, c)
    for a in 1:nv                                   # the vertices are the first nodes of a Lagrangian cell
      g = ids[a]
      g2w[g] == 0 && (np += 1; g2w[g] = np)
      cn[a, c] = g2w[g]
    end
  end
  pts = zeros(3, np)
  for (g, w) in enumerate(g2w)
    w == 0 && continue
    x = X[g]
    for d in 1:D
      pts[d, w] = x[d]
    end
  end
  conn = Vector{Int}(undef, nv * nc)
  for c in 1:nc, a in 1:nv
    conn[(c-1)*nv+a] = cn[perm[a], c] - 1
  end
  cells = [WriteVTK.MeshCell(wt, ntuple(a -> Int(cn[perm[a], c]), nv)) for c in 1:nc]
  WriterMesh(pts, cn, conn, nv, UInt8(wt.vtk_id), cells, get_vertex_coordinates(p))
end

# ---------------------------------------------------------------------------------------------------------------
# Evaluation plans: how to get the values of a field at the nodes (or cells) of the writer mesh

abstract type _Node end

struct _Leaf{F,T} <: _Node         # values gathered from the dof vectors of an FE function
  f::F
  ids::Matrix{Int32}               # ncomp × n; >0 free dof, <0 Dirichlet dof, 0 outside the triangulation of the field
  loc::Symbol                      # :point or :cell
  buf::Matrix{Float64}
end

struct _OpNode{O,A<:Tuple,T} <: _Node   # a pointwise operation on fast nodes
  op::O
  args::A
  loc::Symbol
  buf::Vector{T}
end

struct _ExtNode{N,T} <: _Node      # extend_field: a fixed value outside the triangulation of the field
  child::N
  inside::Vector{Bool}
  outside::T
  loc::Symbol
  buf::Vector{T}
end

struct _LazyNode{F,P,T} <: _Node   # any CellField: evaluated by Gridap at the vertices, averaged at the nodes
  f::F
  pts::P
  cn::Matrix{Int32}
  cnt::Vector{Int32}
  loc::Symbol
  buf::Vector{T}
end

_valtype(::_Leaf{F,T}) where {F,T} = T
_valtype(::_OpNode{O,A,T}) where {O,A,T} = T
_valtype(::_ExtNode{N,T}) where {N,T} = T
_valtype(::_LazyNode{F,P,T}) where {F,P,T} = T

_basis_valtype(::Gridap.ReferenceFEs.LagrangianDofBasis{P,V}) where {P,V} =
  V <: MultiValue ? Gridap.TensorValues.change_eltype(V, Float64) : Float64
_comp(x::MultiValue, k) = x.data[k]
_comp(x, k) = x

function _dof_ids(mesh::WriterMesh, Ω, V, allow_outside)
  c2f = _cell_map_or_nothing(get_triangulation(V), Ω)
  bs = get_data(get_fe_dof_basis(V))
  cd = get_cell_dof_ids(V)
  cache = array_cache(cd)
  b = bs[1]
  b isa Gridap.ReferenceFEs.LagrangianDofBasis || throw(NotFast("not a Lagrangian space"))
  T = _basis_valtype(b)
  K = num_indep_components(T)
  nodes = b.nodes
  n2d = b.node_and_comp_to_dof
  nc, np, nv = ncells(mesh), npoints(mesh), mesh.nv
  outside = "FieldWriter: a field is not defined on every cell of Ω; write it on its own triangulation or use extend_field"
  if length(nodes) == 1                            # one node per cell: cell data
    ids = zeros(Int32, K, nc)
    for c in 1:nc
      fc = c2f === nothing ? c : c2f[c]
      fc == 0 && (allow_outside ? continue : throw(ArgumentError(outside)))
      d = getindex!(cache, cd, fc)
      for k in 1:K
        ids[k, c] = d[_comp(n2d[1], k)]
      end
    end
    return ids, :cell, T
  end
  v2n = [findfirst(x -> sum(abs, Tuple(x - v)) < 1e-12, nodes) for v in mesh.verts]
  any(isnothing, v2n) && throw(NotFast("the space has no nodes at the vertices"))
  ids = zeros(Int32, K, np)
  for c in 1:nc
    fc = c2f === nothing ? c : c2f[c]
    fc == 0 && (allow_outside ? continue : throw(ArgumentError(outside)))
    d = getindex!(cache, cd, fc)
    for a in 1:nv, k in 1:K
      i = d[_comp(n2d[v2n[a]], k)]
      j = mesh.cn[a, c]
      ids[k, j] == 0 || ids[k, j] == i || throw(NotFast("discontinuous space"))
      ids[k, j] = i
    end
  end
  ids, :point, T
end

function _cell_map_or_nothing(Ωf, Ω)
  Ωf === Ω && return nothing
  get_background_model(Ωf) === get_background_model(Ω) || throw(NotFast("the field is on another background model"))
  m = _cell_map(Ωf, Ω)
  m == 1:length(m) && num_cells(Ωf) == length(m) && return nothing
  m
end

function _plan(w, f::Gridap.FESpaces.SingleFieldFEFunction, allow_outside = false)
  V = get_fe_space(f)
  ids, loc, T = get!(w.ids, (objectid(V), allow_outside)) do
    _dof_ids(w.mesh, w.Ω, V, allow_outside)
  end
  _Leaf{typeof(f),T}(f, ids, loc, zeros(size(ids)))
end

function _plan(w, f::Gridap.CellData.OperationCellField, allow_outside = false)
  args = map(a -> _plan(w, a, allow_outside), f.args)
  loc = args[1].loc
  all(a -> a.loc == loc, args) || throw(NotFast("operation on fields at different locations (cell and point)"))
  op = f.op isa Operation ? f.op.op : f.op
  T = typeof(op(map(a -> zero(_valtype(a)), args)...))
  _OpNode{typeof(op),typeof(args),T}(op, args, loc, Vector{T}(undef, loc == :cell ? ncells(w.mesh) : npoints(w.mesh)))
end

function _plan(w, e::ExtendedField, allow_outside = false)
  child = _plan(w, e.f, true)
  c2f = _cell_map_or_nothing(get_triangulation(e.f), w.Ω)
  cin = c2f === nothing ? trues(ncells(w.mesh)) : c2f .> 0
  inside = if child.loc == :cell
    collect(cin)
  else
    v = fill(false, npoints(w.mesh))
    for c in 1:ncells(w.mesh)
      cin[c] && (v[view(w.mesh.cn, :, c)] .= true)  # nodes of the interface take the inside value
    end
    v
  end
  T = _valtype(child)
  _ExtNode{typeof(child),T}(child, inside, _outside(zero(T), e.outside), child.loc, Vector{T}(undef, length(inside)))
end

_plan(w, f, allow_outside = false) = throw(NotFast("no fast path for $(nameof(typeof(f)))"))

function _lazy_plan(w, f)
  m = w.mesh
  pts = Gridap.CellData.CellPoint(Gridap.Arrays.Fill(m.verts, ncells(m)), w.Ω, ReferenceDomain())
  T = typeof(first(f(pts))[1])
  cnt = zeros(Int32, npoints(m))
  for j in m.cn
    cnt[j] += 1
  end
  _LazyNode{typeof(f),typeof(pts),T}(f, pts, m.cn, cnt, :point, Vector{T}(undef, npoints(m)))
end

function _gather!(buf, ids, free, dir)
  @inbounds for i in eachindex(ids)
    d = ids[i]
    buf[i] = d > 0 ? free[d] : d < 0 ? dir[-d] : 0.0
  end
end

function _values!(n::_Leaf{F,T}) where {F,T}
  _gather!(n.buf, n.ids, get_free_dof_values(n.f), get_dirichlet_dof_values(n.f))
  T <: MultiValue ? reinterpret(reshape, T, n.buf) : vec(n.buf)
end
_values!(n::_OpNode) = map!(n.op, n.buf, map(_values!, n.args)...)
function _values!(n::_ExtNode)
  v = _values!(n.child)
  @inbounds for i in eachindex(v)
    n.buf[i] = n.inside[i] ? v[i] : n.outside
  end
  n.buf
end
_values!(n::_LazyNode) = _scatter!(n.buf, n.f(n.pts), n.cn, n.cnt)
function _scatter!(buf::Vector{T}, cv, cn, cnt) where T
  fill!(buf, zero(T))
  cache = array_cache(cv)
  @inbounds for c in 1:size(cn, 2)
    v = getindex!(cache, cv, c)
    for a in 1:size(cn, 1)
      buf[cn[a, c]] += v[a]
    end
  end
  @inbounds for i in eachindex(buf)
    buf[i] = buf[i] / cnt[i]
  end
  buf
end

_ncomp_out(::Type{T}) where {T<:Real} = 1
_ncomp_out(::Type{T}) where {T<:VectorValue} = 3                    # VTK vectors have three components
_ncomp_out(::Type{T}) where {T<:MultiValue} = num_indep_components(T)

function _fill_out!(out::Matrix, v::AbstractVector{T}) where {T<:Real}
  @inbounds for i in eachindex(v)
    out[1, i] = v[i]
  end
end
function _fill_out!(out::Matrix, v::AbstractVector{T}) where {T<:MultiValue}
  K = num_indep_components(T)
  size(out, 1) > K && fill!(out, 0)
  @inbounds for i in eachindex(v), k in 1:K
    out[k, i] = v[i].data[k]
  end
end

# ---------------------------------------------------------------------------------------------------------------
# Writer

struct _Job{R}
  index::Int
  time::Float64
  names::Vector{String}
  locs::Vector{Symbol}
  arrays::Vector{Matrix{R}}
end

"""
    FieldWriter(Ω, base; format = :vtu, precision = Float64, compress = false, async = Threads.nthreads() > 1,
                maxpending = 2, collection = nothing)

Fast writer of fields on the triangulation `Ω` (one cell type, cells of the dimension of the space; a part of a
model is fine). The mesh is built once with the node numbering of the grid (continuous, not one copy of each node per
cell as `createvtk`), and the values come straight from the degrees of freedom:

- P0 fields as cell data; P1/Q1 (and the vertices of higher orders) as point data, scalar or vector, with their
  Dirichlet values; pointwise operations on them (`θ.ρ`, `E ∘ θ.ρ`); [`extend_field`](@ref) (nodes of the interface
  take the inside value).
- Any other `CellField` (discontinuous fields, gradients) is evaluated by Gridap at the vertices and averaged at the
  nodes, in the same file, with a warning once per field (slower).

`format`: `:vtu` (one file `base_<index>.vtu` per step; with `collection`, a [`get_pvd`](@ref) collection, each file is
added at its time once written) or `:vtkhdf` (one file `base.vtkhdf` with every step and the mesh written once; needs
`using HDF5`, ParaView ≥ 5.12). `precision = Float32` halves the values (the points stay `Float64`); `compress` uses
zlib (smaller, slower). With `async` the values are copied and the file is written by another task: the loop waits only
for the copy, at most `maxpending` steps are queued, and an error of a write is thrown by the next
[`write_fields`](@ref) or by `close`. `close(w)` waits for the pending writes (idempotent); `wait(w)` waits without
closing. [`OptPostProcessor`](@ref) builds and closes these writers itself (see [`write_step`](@ref)).
"""
mutable struct FieldWriter{R}
  Ω::Any
  mesh::WriterMesh
  base::String
  format::Symbol
  compress::Bool
  async::Bool
  collection::Any
  ids::Dict{Any,Any}
  plans::Dict{String,Tuple{UInt,_Node}}
  slow::Vector{String}
  step::Int
  created::Bool
  chan::Union{Nothing,Channel{_Job{R}}}
  task::Union{Nothing,Task}
  pool::Channel{Vector{Matrix{R}}}
  pending::Threads.Atomic{Int}
  err::Any
  closed::Bool
end

const FORMATS = (:vtu, :vtkhdf)

function FieldWriter(Ω::Triangulation, base::AbstractString; format::Symbol = :vtu, precision::Type = Float64,
                     compress::Bool = false, async::Bool = Threads.nthreads() > 1, maxpending::Integer = 2,
                     collection = nothing, resume_upto = nothing)
  format in FORMATS || throw(ArgumentError("FieldWriter: format must be one of $(join(FORMATS, ", "))"))
  precision in (Float32, Float64) || throw(ArgumentError("FieldWriter: precision must be Float32 or Float64"))
  maxpending >= 1 || throw(ArgumentError("FieldWriter: maxpending must be at least 1"))
  format === :vtkhdf && !_hdf5_loaded() &&
    throw(ArgumentError("FieldWriter: format = :vtkhdf needs `using HDF5` in the script"))
  base = abspath(base)
  mkpath(dirname(base))
  R = precision
  w = FieldWriter{R}(Ω, WriterMesh(Ω), base, format, compress, async, collection, Dict(), Dict(), String[], 0, false,
                     nothing, nothing, Channel{Vector{Matrix{R}}}(Inf), Threads.Atomic{Int}(0), nothing, false)
  if format === :vtkhdf
    file = base * ".vtkhdf"
    if resume_upto !== nothing && isfile(file)
      w.step = _vtkhdf_truncate(file, resume_upto)   # keep the steps up to the checkpoint and go on appending
      w.created = true
    else
      isfile(file) && rm(file)
    end
  end
  if async
    w.chan = Channel{_Job{R}}(maxpending)
    w.task = Threads.@spawn _writer_loop(w)
  end
  w
end

function _writer_loop(w::FieldWriter)
  for job in w.chan
    try
      w.err === nothing && _write(w, job)
    catch e
      w.err = e
    end
    put!(w.pool, job.arrays)
    Threads.atomic_sub!(w.pending, 1)
  end
end

"""
    write_fields(w::FieldWriter, t, fields::Pair...; index = <step counter>)

Writes `fields` (`"name" => field`) at time `t`. The values are copied before returning; with `async` the file is
written by the writer's task. `index` numbers the VTU file (`base_<index>.vtu`, 4 digits at least).
"""
function write_fields(w::FieldWriter{R}, t::Real, fields::Pair...; index::Integer = w.step) where {R}
  w.closed && throw(ArgumentError("write_fields: the writer is closed"))
  _throw_pending!(w)
  names = String[]
  locs = Symbol[]
  arrays = isready(w.pool) ? take!(w.pool) : Matrix{R}[]
  for (i, (name, f)) in enumerate(fields)
    name = String(name)
    key = objectid(f)
    id, node = get(w.plans, name, (UInt(0), nothing))
    if node === nothing || id != key
      node = try
        _plan(w, f)
      catch e
        e isa NotFast || rethrow()
        if !(name in w.slow)
          push!(w.slow, name)
          @warn "FieldWriter: field \"$name\" is evaluated by Gridap at the vertices ($(e.msg)); slower than the fast path"
        end
        _lazy_plan(w, f)
      end
      w.plans[name] = (key, node)
    end
    v = _values!(node)
    length(arrays) < i && push!(arrays, Matrix{R}(undef, _ncomp_out(eltype(v)), length(v)))
    size(arrays[i]) == (_ncomp_out(eltype(v)), length(v)) || (arrays[i] = Matrix{R}(undef, _ncomp_out(eltype(v)), length(v)))
    _fill_out!(arrays[i], v)
    push!(names, name)
    push!(locs, node.loc)
  end
  length(arrays) > length(fields) && resize!(arrays, length(fields))
  job = _Job{R}(Int(index), Float64(t), names, locs, arrays)
  w.step += 1
  if w.async
    Threads.atomic_add!(w.pending, 1)
    put!(w.chan, job)
  else
    _write(w, job)
    put!(w.pool, arrays)
  end
  nothing
end

function _throw_pending!(w::FieldWriter)
  w.err === nothing && return
  e = w.err
  w.err = nothing
  throw(e)
end

function Base.wait(w::FieldWriter)
  while w.pending[] > 0
    yield()
    sleep(0.0005)
  end
  _throw_pending!(w)
  w
end

function Base.close(w::FieldWriter)
  w.closed && return w
  w.closed = true
  if w.async && w.task !== nothing
    close(w.chan)
    wait(w.task)
    w.task = nothing
  end
  _throw_pending!(w)
  w
end

function _write(w::FieldWriter, job::_Job)
  if w.format === :vtu
    file = string(w.base, "_", lpad(job.index, 4, '0'))
    vtk = WriteVTK.vtk_grid(file, w.mesh.points, w.mesh.cells; append = true, compress = w.compress)
    for (n, l, a) in zip(job.names, job.locs, job.arrays)
      vtk[n, l === :point ? WriteVTK.VTKPointData() : WriteVTK.VTKCellData()] = size(a, 1) == 1 ? vec(a) : a
    end
    files = WriteVTK.vtk_save(vtk)
    w.collection === nothing || (w.collection[job.time] = only(files))   # the collection locks itself
  else
    file = w.base * ".vtkhdf"
    if !w.created
      _vtkhdf_create(file, w.mesh, job, w.compress)
      w.created = true
    end
    _vtkhdf_append(file, job)
  end
  nothing
end

# VTKHDF: methods in ext/HyperShapeHDF5Ext.jl
function _vtkhdf_create end
_hdf5_loaded() = !isempty(methods(_vtkhdf_create))
function _vtkhdf_append end
function _vtkhdf_truncate end
