"""
    PhysicalFields(x::DesignFields; name = map, other = :source => map, ..., backend = :compiled)

Physical fields θ, the ones the weak forms read. Each one comes from one design field through a map (a
[`FieldMap`](@ref), possibly composed with `∘`): `ρ = map` uses the design field `ρ`; `E = :ρ => map` uses the
design field `ρ` under another name, so several physical fields can come from the same design field (their
sensitivities are added). Every design field must be used. The constructor brings the fields up to date once,
calling the solve functions of the operators with `ctx = OptContext(:setup)`. Read a field as `θ.ρ`: a fixed object (an
`FEFunction` or a `CellField`) updated in place, so the weak forms can capture it.

A frozen zone of the design field (see [`FieldVariable`](@ref)) is imposed at the end of every chain: on its cells
θ is the image of the frozen value by the pointwise maps of the chain (`ρ = 1` gives `θ = 1` through a projection),
whatever the filters, and its derivative vanishes there. It costs no solve and keeps a linear chain linear.

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
    map isa FieldMap || throw(ArgumentError("PhysicalFields: field $k: expected a map (Identity(), HelmholtzFilter, TanhProjection, CellMean, PointwiseMap, …), got $(typeof(map))"))
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
    # a frozen zone of the design field: θ takes the image of the frozen value there (the zone is real material)
    zone = get_optvariables(x).blocks[i].zone
    if zone !== nothing
      value = zone.value
      for m in _maps(map)
        _pointwise(m) && (value = output(_stage(m, value)))
      end
      push!(stages, _mask_stage(input, zone, value))
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
changed (`set_beta!`, a `Parameter` of a `PointwiseMap` or of a filter's operator). Note that the fields themselves
read β lazily: forms assembled after `set_beta!` already see
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
    seen_parameters |= has_parameters(s)      # a stage with parameters, or after one, that is not lazy
    seen_parameters && !is_lazy(s) && return true
  end
  false
end

_parameter_version(chain::Tuple) = sum(parameter_version, chain; init = 0)

_update_chain!(chain::Tuple{}, ctx) = nothing
_update_chain!(chain::Tuple, ctx) = (update_stage!(first(chain), ctx); _update_chain!(Base.tail(chain), ctx))

# Reverse sweep through the maps x → θ. A linear form in δθ is pulled back through the lazy stages (projections,
# pointwise maps: the derivative goes inside the integral) until it reaches a stage that is not lazy (filter, cell
# mean), where it is assembled and accumulated with its weight; or until it reaches the design field, where it is
# assembled and added to the design sensitivity. `finish_sink!` then pulls each accumulated vector back once (one
# solve of a filter for all the contributions) and maps the result to z.

_push_back(chain::Tuple{}, L, w, ctx, key) = L
function _push_back(chain::Tuple, L, w, ctx, key)
  s = last(chain)
  is_lazy(s) || (accumulate!(s, L, w, key); return nothing)
  _push_back(Base.front(chain), pullback(s, L, ctx, key), w, ctx, key)
end

# adds w ⋅ (form assembled on the design space of field i)
function _add_to_design!(θ, i, L, w, key)
  gx, gtmp, assems = getfield(θ, :gx), getfield(θ, :gtmp), getfield(θ, :assems)
  assemble_form!(gtmp[i], assems[i], L, key)
  gx[i] .+= w .* gtmp[i]
  nothing
end

# flushes the pending stages of a chain from the last one to the first: what a stage gives back is pushed further
# back with weight one (into an earlier stage that is not lazy, or into the design field)
_flush_chain!(θ, i, chain::Tuple{}, ctx, key) = nothing
function _flush_chain!(θ, i, chain::Tuple, ctx, key)
  s = last(chain)
  front = Base.front(chain)
  if is_pending(s)
    k = (key..., length(chain))
    L = _push_back(front, flush!(s, ctx), 1.0, ctx, k)
    L === nothing || _add_to_design!(θ, i, L, 1.0, k)
  end
  _flush_chain!(θ, i, front, ctx, key)
end

_reset_chain!(chain::Tuple{}) = nothing
_reset_chain!(chain::Tuple) = (reset_accumulator!(first(chain)); _reset_chain!(Base.tail(chain)))

"""
    reset_sink!(θ::PhysicalFields) -> θ

Starts a reverse sweep: the design sensitivities and the accumulators of the filters are set to zero.
"""
function reset_sink!(θ::PhysicalFields)
  foreach(g -> fill!(g, 0.0), getfield(θ, :gx))
  foreach(_reset_chain!, getfield(θ, :chains))
  θ
end

"""
    add_forms!(θ::PhysicalFields, forms::NamedTuple, w, ctx; key = 0) -> θ

Adds `w` times the derivative given by `forms` (`(ρ = δ -> ∫(… * δ)dΩ, …)`, one entry per physical field; missing
fields contribute nothing) to the reverse sweep started by [`reset_sink!`](@ref). Projections and pointwise maps act
inside the integrals now; filters only accumulate, and are solved once by [`finish_sink!`](@ref). `key` identifies
the forms for the compiled backend (see [`design_gradient!`](@ref)).
"""
function add_forms!(θ::PhysicalFields, forms::NamedTuple, w::Real, ctx; key = 0)
  names, sources, chains = getfield(θ, :names), getfield(θ, :sources), getfield(θ, :chains)
  for k in keys(forms)
    k in names || throw(ArgumentError("derivative with respect to θ: no physical field named $k; available: $(join(names, ", "))"))
  end
  for (j, name) in enumerate(names)
    haskey(forms, name) || continue
    L = _push_back(chains[j], forms[name], Float64(w), ctx, (key, j))
    L === nothing || _add_to_design!(θ, sources[j], L, Float64(w), (key, j))
  end
  θ
end

"""
    finish_sink!(gz, θ::PhysicalFields, ctx) -> gz

Ends a reverse sweep: pulls the accumulated sensitivities back through the filters (one transposed solve per filter
with something accumulated), adds them over the physical fields that share a design field, maps them to z and writes
`gz` (a vector of the length of z).
"""
function finish_sink!(gz, θ::PhysicalFields, ctx)
  x = getfield(θ, :x)
  z = get_optvariables(x)
  length(gz) == length(z) || throw(DimensionMismatch("gradient: gz has $(length(gz)) values, z has $(length(z))"))
  sources, chains, gx = getfield(θ, :sources), getfield(θ, :chains), getfield(θ, :gx)
  for (j, chain) in enumerate(chains)
    _flush_chain!(θ, sources[j], chain, ctx, (:flush, j))
  end
  fill!(gz, 0)
  for i in eachindex(gx)
    _pullback_block!(view(gz, z.ranges[i]), x, i, gx[i])
  end
  gz
end

"""
    design_gradient!(gz, θ::PhysicalFields, forms::NamedTuple, ctx; key = 0) -> gz

Derivative with respect to z of a quantity whose derivative with respect to the physical fields is given as linear
forms: `forms = (ρ = δ -> ∫(… * δ)dΩ, …)`, one entry per physical field it depends on (missing fields contribute
nothing). The forms are pulled back through the maps x → θ (transposed filters, derivatives of the projections),
assembled on the design spaces, added over the physical fields that share a design field, and mapped to z.
Writes `gz` (a vector of the length of z). It is [`reset_sink!`](@ref), [`add_forms!`](@ref) with weight one and
[`finish_sink!`](@ref).

`key` identifies the quantity for the compiled backend, which keeps the compiled forms of the last keys used: with
a different key for each quantity (objective, each constraint) the forms are only rebound at every call; forms of
different structure under one key are correct too, their kernels are compiled again at each call (a few
milliseconds). Avoid creating the `Measure` inside a form or capturing a new `FEFunction` at every call: both change
the structure of the form.
"""
function design_gradient!(gz, θ::PhysicalFields, forms::NamedTuple, ctx; key = 0)
  update_fields!(θ, ctx)
  reset_sink!(θ)
  add_forms!(θ, forms, 1.0, ctx; key)
  finish_sink!(gz, θ, ctx)
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
([`Identity`](@ref), [`HelmholtzFilter`](@ref), [`CellMean`](@ref)); a [`TanhProjection`](@ref) or a
[`PointwiseMap`](@ref) is not.
"""
function is_linear_field(θ::PhysicalFields, name::Symbol)
  j = findfirst(==(name), getfield(θ, :names))
  j === nothing && throw(ArgumentError("PhysicalFields: no physical field named $name"))
  all(is_linear_stage, getfield(θ, :chains)[j])
end

Base.show(io::IO, θ::PhysicalFields) = print(io, "PhysicalFields(", join(getfield(θ, :names), ", "), ")")

# for check_consistency: (number of frozen cells, RMS of θ − frozen image over them) of the physical field name, or
# nothing without a frozen zone
function _frozen_check(θ::PhysicalFields, name::Symbol)
  j = findfirst(==(name), getfield(θ, :names))
  s = last(getfield(θ, :chains)[j])
  s isa MaskStage || return nothing
  f = getproperty(θ, name)
  dΩ = Measure(get_triangulation(s.χ), 4)
  area = sum(∫(s.χ)dΩ)
  e = sum(∫(s.χ * (f - s.value) * (f - s.value))dΩ)
  zone = get_optvariables(getfield(θ, :x)).blocks[getfield(θ, :sources)[j]].zone
  zone.ncells, sqrt(max(e, 0.0) / area)
end
