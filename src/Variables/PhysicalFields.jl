"""
    PhysicalFields(x::DesignFields; name = map, other = :source => map, ..., backend = :compiled)

Physical fields θ, the ones the weak forms read. Each one comes from one design field through a map (a
[`FieldMap`](@ref), possibly composed with `∘`): `ρ = map` uses the design field `ρ`; `E = :ρ => map` uses the
design field `ρ` under another name, so several physical fields can come from the same design field (their
sensitivities are added). Every design field must be used. The constructor brings the fields up to date once,
calling the solve functions of the operators with `ctx = OptContext(:setup)`. Read a field as `θ.ρ`: a fixed object (an
`FEFunction` or a `CellField`) updated in place, so the weak forms can capture it.

`backend` (a reserved name: no physical field can be called `backend`) selects who assembles the sensitivities on
the design spaces: `:compiled` (default, HyperFEMKernels) or `:lazy` (Gridap).
"""
struct PhysicalFields{X<:DesignFields,N,C<:Tuple,F<:NamedTuple,A<:Tuple}
  x::X
  names::NTuple{N,Symbol}
  sources::NTuple{N,Int}    # index of the design field of each physical field
  chains::C                 # tuple of tuples of stages
  fields::F
  gx::Vector{Vector{Float64}}   # sensitivity buffers, one per design field
  gtmp::Vector{Vector{Float64}}
  assems::A
  seen::Base.RefValue{Int}          # version of x at the last update
  seen_params::Vector{Int}          # parameter version of each chain at the last update
  version::Base.RefValue{Int}
end

function PhysicalFields(x::DesignFields; backend = :compiled, kwargs...)
  backend isa Symbol || throw(ArgumentError("PhysicalFields: `backend` is a reserved name (it selects :compiled or :lazy) and cannot be the name of a physical field"))
  _check_backend(backend)
  isempty(kwargs) && throw(ArgumentError("PhysicalFields needs at least one field"))
  xnames = getfield(x, :names)
  names = Tuple(keys(kwargs))
  sources = Int[]
  chains = []
  for (k, spec) in pairs(kwargs)
    src, map = spec isa Pair ? (spec.first, spec.second) : (k, spec)
    src isa Symbol || throw(ArgumentError("PhysicalFields: field $k: the source must be a Symbol, got $(typeof(src))"))
    map isa FieldMap || throw(ArgumentError("PhysicalFields: field $k: expected a map (Identity(), HelmholtzFilter, TanhProjection, …), got $(typeof(map))"))
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
  assems = Tuple(form_assembler(backend, TrialFESpace(V), V) for V in spaces)
  θ = PhysicalFields(x, names, Tuple(sources), chains, fields, gx, gtmp, assems, Ref(-1), fill(-1, length(chains)), Ref(0))
  update_fields!(θ, OptContext(:setup))
  θ
end

"""
    state_version(θ::PhysicalFields)

Counter of changes of the physical fields. It grows in [`update_fields!`](@ref) when z or a parameter of a map
(`set_beta!`) changed. Note that the fields themselves read β lazily: forms assembled after `set_beta!` already see
the new value, but operators that depend on θ only notice the change after `update_fields!`. Two `PhysicalFields`
built on the same `DesignFields` are independent: each one is brought up to date by its own `update_fields!`.
"""
state_version(θ::PhysicalFields) = getfield(θ, :version)[]
function Base.getproperty(θ::PhysicalFields, s::Symbol)
  f = getfield(θ, :fields)
  haskey(f, s) || throw(ArgumentError("no physical field named $s; available: $(join(keys(f), ", "))"))
  getfield(f, s)
end
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
  xchanged = seen[] != state_version(x)
  changed = xchanged
  for (j, chain) in enumerate(chains)
    params = _parameter_version(chain)
    pchanged = params != seen_params[j]
    # a new design value recomputes every chain; a new parameter only the chains where a map that is not lazy (a
    # filter) comes after a map with parameters (a projection read lazily needs nothing)
    if xchanged || (pchanged && _needs_update_after_parameters(chain))
      _update_chain!(chain, ctx)
    end
    seen_params[j] = params
    changed |= pchanged
  end
  seen[] = state_version(x)
  changed && (getfield(θ, :version)[] += 1)
  θ
end

function _needs_update_after_parameters(chain::Tuple)
  seen_parameters = false
  for s in chain
    seen_parameters && !is_lazy(s) && return true
    seen_parameters |= has_parameters(s)
  end
  false
end

_parameter_version(chain::Tuple) = sum(parameter_version, chain; init = 0)

_update_chain!(chain::Tuple{}, ctx) = nothing
_update_chain!(chain::Tuple, ctx) = (update_stage!(first(chain), ctx); _update_chain!(Base.tail(chain), ctx))

_pullback_chain(chain::Tuple{}, L, ctx, key) = L
_pullback_chain(chain::Tuple, L, ctx, key) = _pullback_chain(Base.front(chain), pullback(last(chain), L, ctx, key), ctx, key)

"""
    design_gradient!(gz, θ::PhysicalFields, forms::NamedTuple, ctx; key = 0) -> gz

Derivative with respect to z of a quantity whose derivative with respect to the physical fields is given as linear
forms: `forms = (ρ = δ -> ∫(… * δ)dΩ, …)`, one entry per physical field it depends on (missing fields contribute
nothing). The forms are pulled back through the maps x → θ (transposed filters, derivatives of the projections),
assembled on the design spaces, added over the physical fields that share a design field, and mapped to z.
Writes `gz` (a vector of the length of z).

`key` identifies the quantity for the compiled backend, which keeps the compiled forms of the last keys used: with
a different key for each quantity (objective, each constraint) the forms are only rebound at every call; forms of
different structure under one key are correct too, their kernels are compiled again at each call (a few
milliseconds). Avoid creating the `Measure` inside a form or capturing a new `FEFunction` at every call: both change
the structure of the form.
"""
function design_gradient!(gz, θ::PhysicalFields, forms::NamedTuple, ctx; key = 0)
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
    L = _pullback_chain(chains[j], forms[name], ctx, (key, j))
    assemble_form!(gtmp[i], assems[i], L, (key, j))
    gx[i] .+= gtmp[i]
  end
  fill!(gz, 0)
  for i in eachindex(gx)
    _pullback_block!(view(gz, z.ranges[i]), x, i, gx[i])
  end
  gz
end

# the next update_fields! recomputes every chain (filters solved again) and the version grows
invalidate!(θ::PhysicalFields) = (getfield(θ, :seen)[] = -1; θ)

"""
    filter_operators(θ::PhysicalFields) -> Vector

The `LinearOperator`s of the filters in the maps of the physical fields, each one once.
"""
function filter_operators(θ::PhysicalFields)
  ops = Any[]
  for chain in getfield(θ, :chains), s in chain
    s isa FilterStage && !any(o -> o === s.op, ops) && push!(ops, s.op)
  end
  ops
end

"""
    is_linear_field(θ::PhysicalFields, name) -> Bool

Whether the physical field `name` depends linearly on its design field: every map of its chain is linear
([`Identity`](@ref), [`HelmholtzFilter`](@ref), [`CellMean`](@ref)); a [`TanhProjection`](@ref) projection is not.
"""
function is_linear_field(θ::PhysicalFields, name::Symbol)
  j = findfirst(==(name), getfield(θ, :names))
  j === nothing && throw(ArgumentError("PhysicalFields: no physical field named $name"))
  all(is_linear_stage, getfield(θ, :chains)[j])
end

Base.show(io::IO, θ::PhysicalFields) = print(io, "PhysicalFields(", join(getfield(θ, :names), ", "), ")")
