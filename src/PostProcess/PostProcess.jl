"""
    HyperShape.PostProcess

Post-processing of an optimisation, one call per design iteration (the pattern of HyperFEM's `PostProcessor`):

- [`OptPostProcessor`](@ref): calls the user's `driver(post, ctx)` every `every` iterations, keeps the `.pvd`
  collection (the iteration is the time) and writes the history of the run to `history.csv`.
- [`write_step`](@ref): writes the fields of the current iteration with a [`FieldWriter`](@ref) (VTU or VTKHDF, the
  mesh once, values from the degrees of freedom, optionally without blocking the loop).
- [`FieldWriter`](@ref), [`write_fields`](@ref): the writer, usable on its own (also outside an optimisation).
- [`extend_field`](@ref): a field defined on the design zone, on a larger triangulation, with a fixed value outside.
"""
module PostProcess

using Gridap
using Gridap.ReferenceFEs: num_dims, is_n_cube, num_vertices, get_vertex_coordinates, get_polytope
using Gridap.Geometry: get_background_model, get_glue, num_cell_dims, num_point_dims, get_grid, get_reffes,
                       get_node_coordinates, get_cell_node_ids
using Gridap.Arrays: array_cache, getindex!
using Gridap.FESpaces: get_fe_dof_basis, get_cell_dof_ids, get_dirichlet_dof_values, get_fe_space
using Gridap.CellData: CellField, get_triangulation, change_domain, ReferenceDomain, Operation, get_data
import Gridap.CellData: DomainStyle
using Gridap.TensorValues: MultiValue, num_indep_components, VectorValue
using WriteVTK: WriteVTK
import HyperFEM: get_pvd
using ..Operators: OptContext

export OptPostProcessor, write_step, extend_field, get_pvd, FieldWriter, write_fields

# ---------------------------------------------------------------------------------------------------------------
# Collection

"""
    PVDCollection(path)

ParaView collection (`.pvd`) written again, under a temporary name and then renamed, after every entry, so it can be
opened while the optimisation runs and stays valid if it is interrupted. `c[t] = file` adds a file written by any
means (paths are stored relative to the folder of the collection); `c[t] = vtk` saves a WriteVTK file (what
`createvtk` returns) and adds it (with a parallel file, only the `.pvtu`). Several entries at the same time are
written as parts of that time.
"""
mutable struct PVDCollection
  path::String
  times::Vector{Float64}
  files::Vector{String}
  parts::Vector{Int}
  lock::ReentrantLock             # entries also come from the tasks of asynchronous writers
end

PVDCollection(path::AbstractString) = PVDCollection(String(path), Float64[], String[], Int[], ReentrantLock())

Base.length(c::PVDCollection) = length(c.times)

function Base.setindex!(c::PVDCollection, file::AbstractString, t::Real)
  lock(() -> _add!(c, file, t), c.lock)
  c
end

function _add!(c::PVDCollection, file, t)
  tf = Float64(t)
  rel = relpath(abspath(file), dirname(abspath(c.path)))
  any(i -> c.times[i] == tf && c.files[i] == rel, eachindex(c.times)) &&
    throw(ArgumentError("the collection already has $rel at time $(_fmt(tf)): give each file of an iteration its own name"))
  push!(c.parts, count(==(tf), c.times))
  push!(c.times, tf)
  push!(c.files, rel)
  _save(c)
  c
end

function Base.setindex!(c::PVDCollection, vtk, t::Real)
  files = WriteVTK.vtk_save(vtk)
  files isa AbstractString && (files = [files])
  par = filter(f -> endswith(f, ".pvtu"), files)
  for f in (isempty(par) ? files : par)
    c[t] = f
  end
  c
end

_fmt(t::Float64) = isinteger(t) ? string(Int(t)) : string(t)

const _XML = ("&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")
_escape(f) = replace(f, _XML...)
_unescape(f) = replace(f, "&lt;" => "<", "&gt;" => ">", "&quot;" => "\"", "&amp;" => "&")

function _save(c::PVDCollection)
  tmp = c.path * ".tmp"
  open(tmp, "w") do io
    println(io, "<?xml version=\"1.0\" encoding=\"utf-8\"?>")
    println(io, "<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\">")
    println(io, "  <Collection>")
    for (t, p, f) in zip(c.times, c.parts, c.files)
      println(io, "    <DataSet timestep=\"", _fmt(t), "\" part=\"", p, "\" file=\"", _escape(f), "\"/>")
    end
    println(io, "  </Collection>")
    println(io, "</VTKFile>")
  end
  mv(tmp, c.path; force = true)
  c
end

# the entries of an existing collection up to time `upto` (a resumed run); written again at once
function _load(path::AbstractString; upto::Real)
  c = PVDCollection(path)
  if isfile(path)
    for l in eachline(path)
      mt, mf = match(r"timestep=\"([^\"]+)\"", l), match(r"file=\"([^\"]+)\"", l)
      (mt === nothing || mf === nothing) && continue
      t = parse(Float64, mt[1])
      t <= upto || continue
      mp = match(r"part=\"(\d+)\"", l)
      push!(c.times, t)
      push!(c.parts, mp === nothing ? count(==(t), c.times) - 1 : parse(Int, mp[1]))
      push!(c.files, _unescape(mf[1]))
    end
    _save(c)
  end
  c
end

# the result of extend_field: a CellField on Ω that remembers the field and the value outside (for the FieldWriter)
struct ExtendedField{F,O,C<:CellField} <: CellField
  f::F
  outside::O
  cf::C
end
Gridap.CellData.get_data(e::ExtendedField) = get_data(e.cf)
Gridap.CellData.get_triangulation(e::ExtendedField) = get_triangulation(e.cf)
DomainStyle(::Type{<:ExtendedField{F,O,C}}) where {F,O,C} = DomainStyle(C)

include("FieldWriters.jl")

# ---------------------------------------------------------------------------------------------------------------
# OptPostProcessor

"""
    OptPostProcessor(driver; every, filepath = nothing, initial = true, history = true, name = "optimization",
                     format = :vtu, precision = Float64, compress = false, async = Threads.nthreads() > 1,
                     maxpending = 2)

Post-processing of an optimisation: pass it to `solve!(…; post)`. `solve!` calls `driver(post, ctx)` (written
by the user, as the driver of HyperFEM's `PostProcessor`) with the fields, the variables and the forward state at the
accepted design:

- at the starting design (iteration 0, or the one of a resumed checkpoint), if `initial`. It costs no extra solve:
  the forward solve of the first iteration is done before it (so the user's `solve` sees `ctx.iteration == 0` in that
  solve, and the `forward` column of the history counts it in no iteration). Not repeated when the collection
  already has that iteration, nor in a resumed run whose first iteration is a continuation jump;
- after every accepted iteration that is a multiple of `every` (mandatory: one call per iteration can fill the disk;
  the multiples count from iteration 0, also in a resumed run);
- at the end of the run (converged, stalled or `maxiter`) and when it fails (at the last accepted design, if the
  forward problem can still be solved there), when that iteration has not been written yet.

`ctx` is the [`OptContext`](@ref) (phase `:optimization`, `ctx.iteration` the iteration of the design). Nothing is called during [`check_consistency`](@ref
HyperShape.Checks.check_consistency) or for the trial designs of a line search. An error thrown by the driver stops the
optimisation, except in the call after a failure (then it is reported and the failure is thrown).

`filepath` (a folder, created if needed; made absolute when the post-processor is built) enables the collection `filepath/<name>.pvd` ([`get_pvd`](@ref), with the
iteration as time; written after every entry, so it can be opened in ParaView during the run) and, if `history`,
`filepath/history.csv` (the columns of [`OptHistory`](@ref HyperShape.Methods.OptHistory), written again at every call). Without `filepath` the
driver writes what it wants (or nothing: a callback). Each `solve!` starts a new collection (files of an earlier run
in the folder are left there, outside it); a resumed run keeps the entries up to the iteration of the checkpoint.
The collection file is created with its first entry (none with `format = :vtkhdf`).

`format`, `precision`, `compress`, `async` and `maxpending` are the options of the [`FieldWriter`](@ref)s that
[`write_step`](@ref) builds (with `async`, the collection lags behind the loop by the queued steps): `format = :vtkhdf` (one file per triangulation and name with every iteration; needs `using HDF5`) suits large
3D runs, `:vtu` (default) any ParaView. The writers are closed (their pending writes finished) when `solve!` returns,
fails or is interrupted. An error of a write during the run stops it (it is thrown by the next `write_step`); one that
appears when the writers are closed at the end is logged with `@error` and `solve!` still returns its result.
After a stall in a continuation stage whose last iteration was already written, the result of `solve!` is at the new
stage but that iteration is not written again.

The driver can write with [`write_step`](@ref) (HyperShape's fast writer), add any file it writes itself (`get_pvd(post)[ctx.iteration]
= "file.vtu"`) or write nothing to the collection. Fields: `post.filepath`, `post.every`, `post.iteration` (of the
current call) and `post.problem` (the `Optimization` being solved, set by `solve!`; it can be built before or after
the post-processor).

```julia
post = OptPostProcessor(every = 10, filepath = "results/mbb") do post, ctx
  write_step(post, Ω, "rho" => θ.ρ, "x" => x.ρ, "u" => get_state(S))
end
solve!(problem, DensityMethod(OC()); maxiter = 300, post)
```
"""
mutable struct OptPostProcessor{F}
  driver::F
  every::Int
  filepath::Union{Nothing,String}
  initial::Bool
  history::Bool
  name::String
  problem::Any
  iteration::Int
  pvd::Union{Nothing,PVDCollection}
  last::Int                       # last iteration post-processed in the current solve! (-1: none)
  format::Symbol
  precision::Type
  compress::Bool
  async::Bool
  maxpending::Int
  writers::Dict{String,Any}               # the writers of write_step, by name
  resume_upto::Union{Nothing,Int}         # a resumed run: VTKHDF files keep their steps up to this iteration
end

function OptPostProcessor(driver; every::Integer, filepath = nothing, initial::Bool = true, history::Bool = true,
                          name::AbstractString = "optimization", format::Symbol = :vtu, precision::Type = Float64,
                          compress::Bool = false, async::Bool = Threads.nthreads() > 1, maxpending::Integer = 2)
  every >= 1 || throw(ArgumentError("OptPostProcessor: every must be at least 1"))
  filepath === nothing || filepath isa AbstractString || throw(ArgumentError("OptPostProcessor: filepath must be a folder name or nothing"))
  _check_name(name, "OptPostProcessor")
  hasmethod(driver, Tuple{OptPostProcessor{typeof(driver)},OptContext}) ||
    throw(ArgumentError("OptPostProcessor: the driver must take two arguments, driver(post, ctx)"))
  format in FORMATS || throw(ArgumentError("OptPostProcessor: format must be one of $(join(FORMATS, ", "))"))
  precision in (Float32, Float64) || throw(ArgumentError("OptPostProcessor: precision must be Float32 or Float64"))
  maxpending >= 1 || throw(ArgumentError("OptPostProcessor: maxpending must be at least 1"))
  format === :vtkhdf && !_hdf5_loaded() && throw(ArgumentError("OptPostProcessor: format = :vtkhdf needs `using HDF5` in the script"))
  # absolute: a change of directory during the run does not move the output
  OptPostProcessor(driver, Int(every), filepath === nothing ? nothing : abspath(filepath), initial, history, String(name),
                   nothing, -1, nothing, -1, format, precision, compress, async, Int(maxpending), Dict{String,Any}(), nothing)
end

_check_name(name, who) =
  (isempty(name) || occursin('/', name) || occursin('\\', name)) &&
    throw(ArgumentError("$who: name must be a file name, not empty and without path separators (got \"$name\")"))

"""
    get_pvd(post::OptPostProcessor)

The `.pvd` collection of the run (needs `filepath`): `get_pvd(post)[ctx.iteration] = createvtk(…)` or `= "file"`.
"""
function get_pvd(p::OptPostProcessor)
  p.pvd === nothing && throw(ArgumentError(p.filepath === nothing ? "OptPostProcessor: no collection without filepath" :
                                                                   "OptPostProcessor: the collection exists once solve! starts"))
  p.pvd
end

"""
    write_step(post::OptPostProcessor, Ω, fields::Pair...; name = post.name)
    write_step(post::OptPostProcessor, w::FieldWriter, fields::Pair...)

Writes `fields` (`"name" => CellField`) of the current iteration on the triangulation `Ω`, with the
[`FieldWriter`](@ref) of `name` (built at the first call with the `format`, `precision`, `compress`, `async` and
`maxpending` of the post-processor): `post.filepath/<name>_<iteration>.vtu` added to the collection once written, or
`post.filepath/<name>.vtkhdf`. A `name` belongs to one triangulation: several calls in one iteration (other fields,
other triangulations) need different names. The second form writes
with a writer built by the user (the user closes it). For `createvtk` (any field, the discontinuous mesh of Gridap),
write in the driver and add the file: `get_pvd(post)[ctx.iteration] = createvtk(Ω, file; cellfields = …)`.
"""
function write_step(p::OptPostProcessor, Ω::Triangulation, fields::Pair...; name::AbstractString = p.name, kwargs...)
  isempty(kwargs) || throw(ArgumentError("write_step: options ($(join(keys(kwargs), ", "))) are given to OptPostProcessor " *
                                         "(format, precision, compress, async, maxpending)"))
  p.filepath === nothing && throw(ArgumentError("write_step: the OptPostProcessor has no filepath"))
  _check_name(name, "write_step")
  w = get!(p.writers, String(name)) do
    FieldWriter(Ω, joinpath(p.filepath, name); format = p.format, precision = p.precision, compress = p.compress,
                async = p.async, maxpending = p.maxpending, collection = p.format === :vtu ? get_pvd(p) : nothing,
                resume_upto = p.resume_upto)
  end
  w.Ω === Ω || throw(ArgumentError("write_step: name \"$name\" is already written on another triangulation; give each triangulation its own name"))
  write_fields(w, p.iteration, fields...; index = p.iteration)
end

write_step(p::OptPostProcessor, w::FieldWriter, fields::Pair...) = write_fields(w, p.iteration, fields...; index = p.iteration)

# called by solve! at the start of each run: k0 is the starting iteration, a resumed run keeps its entries up to k0
function _post_start!(p::OptPostProcessor, problem, k0::Integer, resumed::Bool)
  _post_close!(p; quiet = true)                # writers left by an earlier run
  p.problem = problem
  p.iteration = k0
  p.last = -1
  p.pvd = nothing
  p.resume_upto = resumed ? Int(k0) : nothing
  if p.filepath !== nothing
    mkpath(p.filepath)
    path = joinpath(p.filepath, p.name * ".pvd")
    if resumed
      p.pvd = _load(path; upto = k0)
      any(==(Float64(k0)), p.pvd.times) && (p.last = k0)     # the starting design is already written
    else
      isfile(path) && rm(path)                   # created again with the first entry of this run
      p.pvd = PVDCollection(path)
    end
  end
  p
end

# called by solve! when it returns (errors of the writers are thrown) and in its `finally` (quiet: reported only)
function _post_close!(p::OptPostProcessor; quiet::Bool = false)
  err = nothing
  try
    for w in values(p.writers)
      try
        close(w)
      catch e
        e isa InterruptException && rethrow()
        err === nothing && (err = e)
      end
    end
  finally
    empty!(p.writers)
  end
  if err !== nothing
    quiet ? (@error "OptPostProcessor: a write failed" exception = err) : throw(err)
  end
  nothing
end

_post_due(p::OptPostProcessor, k::Integer) = k % p.every == 0 && k != p.last

function _post_call!(p::OptPostProcessor, ctx::OptContext)
  p.iteration = ctx.iteration
  p.driver(p, ctx)
  p.last = ctx.iteration
  nothing
end

# ---------------------------------------------------------------------------------------------------------------
# extend_field

struct _Select{T} <: Function
  outside::T
end
(s::_Select)(inside, v) = inside ? v : _outside(v, s.outside)

_outside(v, o) = convert(typeof(v), o)
# a scalar outside a vector or tensor field: the value of every independent component
_outside(v::MultiValue, o::Real) = typeof(v)(ntuple(_ -> convert(eltype(v), o), Val(num_indep_components(typeof(v)))))

# cell of Ω → cell of Ωd (0 outside Ωd)
function _cell_map(Ωd, Ω)
  get_background_model(Ωd) === get_background_model(Ω) ||
    throw(ArgumentError("extend_field: the field and Ω are not on the same background model"))
  D = num_cell_dims(Ω)
  num_cell_dims(Ωd) == D || throw(ArgumentError("extend_field: the field and Ω have different dimensions"))
  gd, g = get_glue(Ωd, Val(D)), get_glue(Ω, Val(D))
  m2d = zeros(Int32, num_cells(get_background_model(Ω)))
  m2d[gd.tface_to_mface] .= 1:num_cells(Ωd)
  m2d[g.tface_to_mface]
end

"""
    extend_field(f::CellField, Ω::Triangulation; outside) -> CellField

The field `f`, defined on a triangulation of the design zone (for example `θ.ρ`, `x.ρ` or a state on that zone), as
a field on the larger triangulation `Ω` of the same background model: equal to `f` on the cells of the design zone and
to `outside` on the others (for example `1.0` for a solid support, `0.0` for air). For a vector or tensor field,
`outside` is a value of that type or a number for every independent component (`0.0` gives zero). For writing (also
with [`write_step`](@ref)) and integrating; returns `f` when `Ω` is its triangulation. The operation that defines `f`
is also evaluated outside, at a zero value of its arguments (it must not throw there). Costs about 70 bytes per cell
of `Ω` per call.
"""
function extend_field(f::CellField, Ω::Triangulation; outside)
  Ωd = get_triangulation(f)
  Ωd === Ω && return f
  inside = CellField(_cell_map(Ωd, Ω) .> 0, Ω)
  ExtendedField(f, outside, Operation(_Select(outside))(inside, change_domain(f, Ω, ReferenceDomain())))
end


end
