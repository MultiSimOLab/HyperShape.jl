"""
    PhysicalFields(x::DesignFields; name = map, other = :source => map, ...)

Physical fields θ, the ones the weak forms read. Each one comes from one design field through a map (a
[`FieldMap`](@ref), possibly composed with `∘`): `ρ = map` uses the design field `ρ`; `E = :ρ => map` uses the
design field `ρ` under another name, so several physical fields can come from the same design field (their
sensitivities are added). Every design field must be used. Read a field as `θ.ρ`: a fixed object (an `FEFunction` or
a `CellField`) updated in place, so the weak forms can capture it.
"""
struct PhysicalFields{X<:DesignFields,N,C<:Tuple,F<:NamedTuple}
  x::X
  names::NTuple{N,Symbol}
  sources::NTuple{N,Int}    # index of the design field of each physical field
  chains::C                 # tuple of tuples of stages
  fields::F
  gx::Vector{Vector{Float64}}   # sensitivity buffers, one per design field
  gtmp::Vector{Vector{Float64}}
  assems::Tuple
  seen::Base.RefValue{Int}          # version of x at the last update
  seen_params::Base.RefValue{Int}   # sum of the parameter versions of the maps at the last update
  version::Base.RefValue{Int}
end

function PhysicalFields(x::DesignFields; kwargs...)
  isempty(kwargs) && throw(ArgumentError("PhysicalFields needs at least one field"))
  xnames = getfield(x, :names)
  names = Tuple(keys(kwargs))
  sources = Int[]
  chains = []
  for (k, spec) in pairs(kwargs)
    src, map = spec isa Pair ? (spec.first, spec.second) : (k, spec)
    src isa Symbol || throw(ArgumentError("PhysicalFields: field $k: the source must be a Symbol, got $(typeof(src))"))
    map isa FieldMap || throw(ArgumentError("PhysicalFields: field $k: expected a map (Identity(), HelmholtzFilter, Heaviside, …), got $(typeof(map))"))
    i = findfirst(==(src), xnames)
    i === nothing && throw(ArgumentError("PhysicalFields: field $k: no design field named $src; available: $(join(xnames, ", "))"))
    push!(sources, i)
    input = getfield(getfield(x, :fields), src)
    stages = []
    for m in _maps(map)
      s = _stage(m, input)
      push!(stages, s)
      input = output(s)
    end
    push!(chains, Tuple(stages))
  end
  unused = [xnames[i] for i in eachindex(xnames) if !(i in sources)]
  isempty(unused) || throw(ArgumentError("PhysicalFields: the design fields $(join(unused, ", ")) are not used by any physical field"))
  chains = Tuple(chains)
  fields = NamedTuple{names}(Tuple(output(last(c)) for c in chains))
  spaces = getfield(x, :spaces)
  gx = [zeros(num_free_dofs(V)) for V in spaces]
  gtmp = [zeros(num_free_dofs(V)) for V in spaces]
  assems = Tuple(SparseMatrixAssembler(V, V) for V in spaces)
  θ = PhysicalFields(x, names, Tuple(sources), chains, fields, gx, gtmp, assems, Ref(-1), Ref(-1), Ref(0))
  update_fields!(θ, nothing)
  θ
end

"""
    version(θ::PhysicalFields)

Counter of changes of the physical fields. It grows in [`update_fields!`](@ref) when z or a parameter of a map
(`set_beta!`) changed. Note that the fields themselves read β lazily: forms assembled after `set_beta!` already see
the new value, but operators that depend on θ only notice the change after `update_fields!`.
"""
version(θ::PhysicalFields) = getfield(θ, :version)[]
Base.getproperty(θ::PhysicalFields, s::Symbol) = getfield(getfield(θ, :fields), s)
Base.propertynames(θ::PhysicalFields) = getfield(θ, :names)
get_designfields(θ::PhysicalFields) = getfield(θ, :x)

"""
    update_fields!(θ::PhysicalFields, ctx)

Brings the design and physical fields up to date with the [`OptVariables`](@ref) and with the parameters of the maps
(filters solved again, version increased); does nothing if nothing changed. `ctx` is passed to the solve functions of the operators.
"""
function update_fields!(θ::PhysicalFields, ctx)
  x = getfield(θ, :x)
  update_fields!(x, ctx)
  seen, seen_params = getfield(θ, :seen), getfield(θ, :seen_params)
  chains = getfield(θ, :chains)
  params = sum(c -> _parameter_version(c), chains)
  seen[] == version(x) && seen_params[] == params && return θ
  for chain in chains
    _update_chain!(chain, ctx)
  end
  seen[] = version(x)
  seen_params[] = params
  getfield(θ, :version)[] += 1
  θ
end

_parameter_version(chain::Tuple) = sum(parameter_version, chain; init = 0)

_update_chain!(chain::Tuple{}, ctx) = nothing
_update_chain!(chain::Tuple, ctx) = (update_stage!(first(chain), ctx); _update_chain!(Base.tail(chain), ctx))

_pullback_chain(chain::Tuple{}, L, ctx) = L
_pullback_chain(chain::Tuple, L, ctx) = _pullback_chain(Base.front(chain), pullback(last(chain), L, ctx), ctx)

"""
    design_gradient!(gz, θ::PhysicalFields, forms::NamedTuple, ctx) -> gz

Derivative with respect to z of a quantity whose derivative with respect to the physical fields is given as linear
forms: `forms = (ρ = δ -> ∫(… * δ)dΩ, …)`, one entry per physical field it depends on (missing fields contribute
nothing). The forms are pulled back through the maps x → θ (transposed filters, derivatives of the projections),
assembled on the design spaces, added over the physical fields that share a design field, and mapped to z.
Writes `gz` (a vector of the length of z).
"""
function design_gradient!(gz, θ::PhysicalFields, forms::NamedTuple, ctx)
  update_fields!(θ, ctx)
  x = getfield(θ, :x)
  z = get_optvariables(x)
  length(gz) == length(z) || throw(DimensionMismatch("design_gradient!: gz has $(length(gz)) values, z has $(length(z))"))
  names = getfield(θ, :names)
  for k in keys(forms)
    k in names || throw(ArgumentError("design_gradient!: no physical field named $k; available: $(join(names, ", "))"))
  end
  gx, gtmp, assems = getfield(θ, :gx), getfield(θ, :gtmp), getfield(θ, :assems)
  spaces = getfield(x, :spaces)
  sources, chains = getfield(θ, :sources), getfield(θ, :chains)
  foreach(g -> fill!(g, 0), gx)
  for (j, name) in enumerate(names)
    haskey(forms, name) || continue
    i = sources[j]
    L = _pullback_chain(chains[j], forms[name], ctx)
    fill!(gtmp[i], 0)
    assemble_vector!(L, gtmp[i], assems[i], spaces[i])
    gx[i] .+= gtmp[i]
  end
  fill!(gz, 0)
  for i in eachindex(gx)
    _pullback_block!(view(gz, z.ranges[i]), x, i, gx[i])
  end
  gz
end
