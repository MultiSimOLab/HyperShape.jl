"""
    DesignFields(z::OptVariables; name = Identity(), ...)

Design fields x built from the blocks of `z`, one per block and with the same name. Each map z → x is mandatory;
only [`Identity`](@ref) is available for now. Read a field as `x.ρ` (a Gridap `FEFunction`, updated in place).
"""
struct DesignFields{Z<:OptVariables,N,F<:NamedTuple,S<:Tuple}
  z::Z
  names::NTuple{N,Symbol}
  fields::F
  spaces::S
  seen::Base.RefValue{Int}
  version::Base.RefValue{Int}
end

function DesignFields(z::OptVariables; kwargs...)
  names = Tuple(keys(kwargs))
  nomap = setdiff(z.names, names)
  isempty(nomap) || throw(ArgumentError("DesignFields: no map for the blocks $(join(nomap, ", ")) (write Identity())"))
  extra = setdiff(names, z.names)
  isempty(extra) || throw(ArgumentError("DesignFields: no block named $(join(extra, ", ")) in the OptVariables"))
  for (k, m) in pairs(kwargs)
    m isa Identity || throw(ArgumentError("DesignFields: map of $k: only Identity() is available, got $(typeof(m))"))
  end
  # fields in the order of the blocks of z
  spaces = Tuple(b.space for b in z.blocks)
  fields = NamedTuple{z.names}(Tuple(FEFunction(b.space, copy(b.init)) for b in z.blocks))
  x = DesignFields(z, z.names, fields, spaces, Ref(-1), Ref(0))
  update_fields!(x, OptContext(:setup))
  x
end

version(x::DesignFields) = getfield(x, :version)[]
function Base.getproperty(x::DesignFields, s::Symbol)
  f = getfield(x, :fields)
  haskey(f, s) || throw(ArgumentError("no design field named $s; available: $(join(keys(f), ", "))"))
  getfield(f, s)
end
Base.propertynames(x::DesignFields) = getfield(x, :names)
get_optvariables(x::DesignFields) = getfield(x, :z)

"""
    update_fields!(x::DesignFields, ctx)

Brings the design fields up to date with their [`OptVariables`](@ref); does nothing if `z` has not changed.
"""
function update_fields!(x::DesignFields, ctx)
  z = getfield(x, :z)
  seen = getfield(x, :seen)
  seen[] == version(z) && return x
  fields = getfield(x, :fields)
  for (i, f) in enumerate(fields)
    copyto!(get_free_dof_values(f), view(z.values, z.ranges[i]))
  end
  seen[] = version(z)
  getfield(x, :version)[] += 1
  x
end

# g_z of block i from g_x of field i (identity map)
_pullback_block!(gz_block, x::DesignFields, i, gx) = (gz_block .+= gx; gz_block)
