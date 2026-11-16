"""
    HyperShape.PostProcess

Post-processing of an optimisation, one call per design iteration (the pattern of HyperFEM's `PostProcessor`):

- [`OptPostProcessor`](@ref): calls the user's `driver(post, ctx)` every `every` iterations, keeps the `.pvd`
  collection (the iteration is the time) and writes the history of the run to `history.csv`.
- [`write_step`](@ref): writes the fields of the current iteration with `createvtk` and adds them to the collection.
- [`extend`](@ref): a field defined on the design zone, on a larger triangulation, with a fixed value outside.
"""
module PostProcess

using Gridap
using Gridap.Geometry: get_background_model, get_glue, num_cell_dims
using Gridap.CellData: CellField, get_triangulation, change_domain, ReferenceDomain, Operation
using WriteVTK: WriteVTK
import HyperFEM: get_pvd
using ..Operators: OptContext

export OptPostProcessor, write_step, extend, get_pvd

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
end

PVDCollection(path::AbstractString) = PVDCollection(String(path), Float64[], String[], Int[])

Base.length(c::PVDCollection) = length(c.times)

function Base.setindex!(c::PVDCollection, file::AbstractString, t::Real)
  tf = Float64(t)
  push!(c.parts, count(==(tf), c.times))
  push!(c.times, tf)
  push!(c.files, relpath(abspath(file), dirname(abspath(c.path))))
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

function _save(c::PVDCollection)
  tmp = c.path * ".tmp"
  open(tmp, "w") do io
    println(io, "<?xml version=\"1.0\" encoding=\"utf-8\"?>")
    println(io, "<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\">")
    println(io, "  <Collection>")
    for (t, p, f) in zip(c.times, c.parts, c.files)
      println(io, "    <DataSet timestep=\"", _fmt(t), "\" part=\"", p, "\" file=\"", f, "\"/>")
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
      m = match(r"timestep=\"([^\"]+)\"\s+part=\"(\d+)\"\s+file=\"([^\"]+)\"", l)
      m === nothing && continue
      t = parse(Float64, m[1])
      t <= upto || continue
      push!(c.times, t)
      push!(c.parts, parse(Int, m[2]))
      push!(c.files, m[3])
    end
  end
  _save(c)
end

# ---------------------------------------------------------------------------------------------------------------
# OptPostProcessor

"""
    OptPostProcessor(driver; every, filepath = nothing, initial = true, history = true, name = "optimization")

Post-processing of an optimisation: pass it to `solve!(…; post)`. `solve!` calls `driver(post, ctx)` (written
by the user, as the driver of HyperFEM's `PostProcessor`) with the fields, the variables and the forward state at the
accepted design:

- at the starting design (iteration 0, or the one of a resumed checkpoint), if `initial` (costs no extra solve: the
  forward solve of the first iteration is done before it);
- after every accepted iteration that is a multiple of `every` (mandatory: one call per iteration can fill the disk);
- at the end of the run (converged, stalled or `maxiter`) and when it fails (at the last accepted design, if the
  forward problem can still be solved there), when that iteration has not been written yet.

`ctx` is the [`OptContext`](@ref) (phase `:optimization`, `ctx.iteration` the iteration of the design). The adjoint
state is still the one of the previous design. Nothing is called during [`check_consistency`](@ref
HyperShape.Checks.check_consistency) or for the trial designs of a line search. An error thrown by the driver stops the
optimisation, except in the call after a failure (then it is reported and the failure is thrown).

`filepath` (a folder, created if needed) enables the collection `filepath/<name>.pvd` ([`get_pvd`](@ref), with the
iteration as time; written after every entry, so it can be opened in ParaView during the run) and, if `history`,
`filepath/history.csv` (the columns of [`OptHistory`](@ref HyperShape.Methods.OptHistory), written again at every call). Without `filepath` the
driver writes what it wants (or nothing: a callback). Each `solve!` starts a new collection; a resumed run keeps the
entries up to the iteration of the checkpoint.

The driver can write with [`write_step`](@ref) (`createvtk`), add any file it writes itself (`get_pvd(post)[ctx.iteration]
= "file.vtu"`) or write nothing to the collection. Fields: `post.filepath`, `post.every`, `post.iteration` (of the
current call) and `post.problem` (the `Optimization` being solved, set by `solve!`; it can be built before or after
the post-processor).

```julia
post = OptPostProcessor(every = 10, filepath = "results/mbb") do post, ctx
  write_step(post, Ω, "rho" => θ.ρ, "x" => x.ρ, "u" => get_state(S))
end
solve!(problem, SIMP(OC()); maxiter = 300, post)
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
end

function OptPostProcessor(driver; every::Integer, filepath = nothing, initial::Bool = true, history::Bool = true,
                          name::AbstractString = "optimization")
  every >= 1 || throw(ArgumentError("OptPostProcessor: every must be at least 1"))
  filepath === nothing || filepath isa AbstractString || throw(ArgumentError("OptPostProcessor: filepath must be a folder name or nothing"))
  isempty(name) && throw(ArgumentError("OptPostProcessor: name must not be empty"))
  OptPostProcessor(driver, Int(every), filepath === nothing ? nothing : String(filepath), initial, history, String(name),
                   nothing, -1, nothing, -1)
end

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
    write_step(post::OptPostProcessor, Ω, fields::Pair...; name = post.name, kwargs...)

Writes `fields` (`"name" => CellField`) on the triangulation `Ω` with `createvtk` to
`post.filepath/<name>_<iteration>` and adds it to the collection at the current iteration. `kwargs` go to
`createvtk` (`compress = true` for smaller files, `order`, `nsubcells`, `celldata`). For other formats, write the
file in the driver and add it with `get_pvd(post)[ctx.iteration] = file`. Several calls in one iteration (different
`Ω` and `name`) are parts of the same time.
"""
function write_step(p::OptPostProcessor, Ω, fields::Pair...; name::AbstractString = p.name, kwargs...)
  p.filepath === nothing && throw(ArgumentError("write_step: the OptPostProcessor has no filepath"))
  file = joinpath(p.filepath, string(name, "_", lpad(p.iteration, 4, '0')))
  get_pvd(p)[p.iteration] = createvtk(Ω, file; cellfields = collect(Pair{String,Any}, (String(f.first) => f.second for f in fields)),
                                      kwargs...)
  nothing
end

# called by solve! at the start of each run: k0 is the starting iteration, a resumed run keeps its entries up to k0
function _post_start!(p::OptPostProcessor, problem, k0::Integer, resumed::Bool)
  p.problem = problem
  p.iteration = k0
  p.last = -1
  p.pvd = nothing
  if p.filepath !== nothing
    mkpath(p.filepath)
    path = joinpath(p.filepath, p.name * ".pvd")
    p.pvd = resumed ? _load(path; upto = k0) : _save(PVDCollection(path))
    resumed && any(==(Float64(k0)), p.pvd.times) && (p.last = k0)     # the starting design is already written
  end
  p
end

_post_due(p::OptPostProcessor, k::Integer) = k % p.every == 0 && k != p.last

function _post_call!(p::OptPostProcessor, ctx::OptContext)
  p.iteration = ctx.iteration
  p.driver(p, ctx)
  p.last = ctx.iteration
  nothing
end

# ---------------------------------------------------------------------------------------------------------------
# extend

struct _Select{T} <: Function
  outside::T
end
(s::_Select)(inside, v) = inside ? v : oftype(v, s.outside)

# cell of Ω → cell of Ωd (0 outside Ωd)
function _cell_map(Ωd, Ω)
  get_background_model(Ωd) === get_background_model(Ω) ||
    throw(ArgumentError("extend: the field and Ω are not on the same background model"))
  D = num_cell_dims(Ω)
  num_cell_dims(Ωd) == D || throw(ArgumentError("extend: the field and Ω have different dimensions"))
  gd, g = get_glue(Ωd, Val(D)), get_glue(Ω, Val(D))
  m2d = zeros(Int32, num_cells(get_background_model(Ω)))
  m2d[gd.tface_to_mface] .= 1:num_cells(Ωd)
  m2d[g.tface_to_mface]
end

"""
    extend(f::CellField, Ω::Triangulation; outside) -> CellField

The field `f`, defined on a triangulation of the design zone (for example `θ.ρ`, `x.ρ` or a state on that zone), as
a field on the larger triangulation `Ω` of the same background model: equal to `f` on the cells of the design zone and
to the scalar `outside` on the others (for example `1.0` for a solid support, `0.0` for air). For writing (also
with [`write_step`](@ref)) and integrating; returns `f` when `Ω` is its triangulation. The operation that defines `f`
is also evaluated outside, at a zero value of its arguments (it must not throw there). Costs about 75 bytes per cell
of `Ω` per call.
"""
function extend(f::CellField, Ω::Triangulation; outside::Number)
  Ωd = get_triangulation(f)
  Ωd === Ω && return f
  inside = CellField(_cell_map(Ωd, Ω) .> 0, Ω)
  Operation(_Select(outside))(inside, change_domain(f, Ω, ReferenceDomain()))
end

end
