# VTKHDF output of HyperShape's FieldWriter (format = :vtkhdf): a temporal UnstructuredGrid (VTKHDF 2.0) with the
# mesh written once and the fields appended at every step. Read by ParaView ≥ 5.12.
module HyperShapeHDF5Ext

using HDF5
using HyperShape.PostProcess: WriterMesh, _Job, npoints, ncells
import HyperShape.PostProcess: _vtkhdf_create, _vtkhdf_append, _vtkhdf_truncate

function _fixed_string_attr(g, name, s)
  dt = HDF5.Datatype(HDF5.API.h5t_copy(HDF5.API.H5T_C_S1))
  HDF5.API.h5t_set_size(dt, length(s))
  HDF5.API.h5t_set_cset(dt, HDF5.API.H5T_CSET_ASCII)
  a = create_attribute(g, name, dt, HDF5.dataspace(HDF5.API.h5s_create(HDF5.API.H5S_SCALAR)))
  HDF5.API.h5a_write(a, dt, Vector{UInt8}(s))
  close(a)
end

_ext1(g, name, T, chunk; kw...) = create_dataset(g, name, datatype(T), dataspace((0,), max_dims = (-1,)); chunk = (chunk,), kw...)
_ext2(g, name, T, K, chunk; kw...) =
  create_dataset(g, name, datatype(T), dataspace((K, 0), max_dims = (K, -1)); chunk = (K, chunk), kw...)

function _append1!(d, v)
  n = length(d)
  HDF5.set_extent_dims(d, (n + length(v),))
  d[n+1:n+length(v)] = v
  n
end

function _append2!(d, v)
  n = size(d, 2)
  HDF5.set_extent_dims(d, (size(d, 1), n + size(v, 2)))
  d[:, n+1:n+size(v, 2)] = v
  n
end

_grp(l) = l === :point ? "PointData" : "CellData"

function _vtkhdf_create(file, m::WriterMesh, job::_Job{R}, compress) where {R}
  np, nc = npoints(m), ncells(m)
  kw = compress ? (; deflate = 4, shuffle = true) : (;)
  h5open(file, "w") do h
    g = create_group(h, "VTKHDF")
    attributes(g)["Version"] = [2, 0]
    _fixed_string_attr(g, "Type", "UnstructuredGrid")
    g["NumberOfPoints"] = [np]
    g["NumberOfCells"] = [nc]
    g["NumberOfConnectivityIds"] = [length(m.conn)]
    g["Points"] = m.points                         # always Float64
    g["Connectivity"] = m.conn
    g["Offsets"] = collect(0:m.nv:length(m.conn))
    g["Types"] = fill(m.ctype, nc)
    pd, cd = create_group(g, "PointData"), create_group(g, "CellData")
    s = create_group(g, "Steps")
    attributes(s)["NSteps"] = 0
    _ext1(s, "Values", Float64, 64)
    for n in ("PartOffsets", "NumberOfParts", "PointOffsets")
      _ext1(s, n, Int, 64)
    end
    for n in ("CellOffsets", "ConnectivityIdOffsets")
      _ext2(s, n, Int, 1, 64)
    end
    po, co = create_group(s, "PointDataOffsets"), create_group(s, "CellDataOffsets")
    for (n, l, a) in zip(job.names, job.locs, job.arrays)
      K, N = size(a)
      grp = l === :point ? pd : cd
      K == 1 ? _ext1(grp, n, R, N; kw...) : _ext2(grp, n, R, K, N; kw...)
      _ext1(l === :point ? po : co, n, Int, 64)
    end
  end
  nothing
end

# one step, all or nothing: if a write fails, every dataset goes back to its size before the step
function _vtkhdf_append(file, job::_Job)
  h5open(file, "r+") do h
    g = h["VTKHDF"]
    s = g["Steps"]
    k = read(attributes(s)["NSteps"])
    touched = Tuple{Any,Tuple}[]
    grow!(d) = push!(touched, (d, size(d)))
    try
      for (n, l, a) in zip(job.names, job.locs, job.arrays)
        d = g[_grp(l)*"/"*n]
        grow!(d)
        off = size(a, 1) == 1 ? _append1!(d, vec(a)) : _append2!(d, a)
        o = s[_grp(l)*"Offsets/"*n]
        grow!(o)
        _append1!(o, [off])
      end
      for nm in ("Values", "PartOffsets", "NumberOfParts", "PointOffsets", "CellOffsets", "ConnectivityIdOffsets")
        grow!(s[nm])
      end
      _append1!(s["Values"], [job.time])
      _append1!(s["PartOffsets"], [0])
      _append1!(s["NumberOfParts"], [1])
      _append1!(s["PointOffsets"], [0])
      _append2!(s["CellOffsets"], reshape([0], 1, 1))
      _append2!(s["ConnectivityIdOffsets"], reshape([0], 1, 1))
      HDF5.delete_attribute(s, "NSteps")
      attributes(s)["NSteps"] = k + 1
    catch
      for (d, dims) in touched
        try
          HDF5.set_extent_dims(d, dims)
        catch
        end
      end
      rethrow()
    end
  end
  nothing
end

# a resumed run: check that the file is of this mesh, keep the steps with time ≤ upto; returns their number, the fields
# (sorted (name, loc, size per step)) and the last time kept
function _vtkhdf_truncate(file, upto, np, nc)
  h5open(file, "r+") do h
    g = h["VTKHDF"]
    (read(g["NumberOfPoints"])[1] == np && read(g["NumberOfCells"])[1] == nc) ||
      throw(ArgumentError("FieldWriter: $(file) is of another mesh; remove it or write elsewhere"))
    s = g["Steps"]
    vals = read(s["Values"])
    n = count(<=(upto), vals)
    issorted(vals) || throw(ArgumentError("FieldWriter: the times of $(file) are not increasing"))
    schema = Tuple{String,Symbol,Tuple{Int,Int}}[]
    for (grpname, N, loc) in (("PointData", np, :point), ("CellData", nc, :cell)), name in keys(g[grpname])
      d = g[grpname*"/"*name]
      dims = size(d)
      K = length(dims) == 1 ? 1 : dims[1]
      push!(schema, (name, loc, (K, N)))
      HDF5.set_extent_dims(d, length(dims) == 1 ? (n * N,) : (K, n * N))
      HDF5.set_extent_dims(s[grpname*"Offsets/"*name], (n,))
    end
    for nm in ("Values", "PartOffsets", "NumberOfParts", "PointOffsets")
      HDF5.set_extent_dims(s[nm], (n,))
    end
    for nm in ("CellOffsets", "ConnectivityIdOffsets")
      HDF5.set_extent_dims(s[nm], (1, n))
    end
    HDF5.delete_attribute(s, "NSteps")
    attributes(s)["NSteps"] = n
    n, sort!(schema), n == 0 ? -Inf : vals[n]
  end
end

end
