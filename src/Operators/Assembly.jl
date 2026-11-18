# Assembly of the vectors of linear forms (load vector of a filter, sensitivities), by one of two backends:
#   :compiled  HyperFEMKernels: the form is compiled once into threaded kernels and re-evaluated at every call
#   :lazy      Gridap's assembler
#
# A compiled assembler keeps the kernel operator of the forms it assembled, one per `key`, up to a capacity
# (`MAX_COMPILED_FORMS` at first): beyond it the least recently used is dropped (never `:forward`, the load vector of
# a filter). A key that comes back after being dropped raises the capacity of that assembler by one (up to
# `MAX_CAPACITY`, with one warning there), so a problem with many functionals stops recompiling after a few iterations.
# All of them share tables, geometry and colouring: the operator of a new key is derived from an existing one. A form assembled again under its key with the same structure (same
# measures, operations and fields, with new values) only rebinds values; with another structure its kernels are
# compiled again, which is correct and costs a few milliseconds. Two things change the structure at every call and
# should be avoided in forms that are assembled repeatedly: creating the `Measure` inside the form and capturing a
# new `FEFunction` each time (keep one and update its values).

abstract type FormAssembler end

struct LazyAssembler{A,S} <: FormAssembler
  assem::A
  V::S
end

struct CompiledAssembler{T,S,M} <: FormAssembler
  U::T
  V::S
  dΩ::M                    # for the (never assembled) tangent the kernel operator asks for
  x0::Vector{Float64}
  keys::Vector{Any}        # keys and their kernel operators, least recently used first
  ops::Vector{Any}
  capacity::Base.RefValue{Int}
  evicted::Vector{Any}     # keys dropped so far (a key that comes back raises the capacity)
  warned::Base.RefValue{Bool}
end

const MAX_COMPILED_FORMS = 16
const MAX_CAPACITY = 256

const BACKENDS = (:compiled, :lazy)

function _check_backend(backend)
  backend in BACKENDS || throw(ArgumentError("backend must be :compiled or :lazy, got $(repr(backend))"))
  backend
end

function form_assembler(backend::Symbol, U, V)
  _check_backend(backend)
  # multi-field test spaces (the adjoint of a coupled state) are assembled by Gridap
  (backend === :lazy || V isa MultiFieldFESpace) && return LazyAssembler(SparseMatrixAssembler(U, V), V)
  CompiledAssembler(U, V, Measure(get_triangulation(V), 1), zeros(num_free_dofs(U)), Any[], Any[],
                    Ref(MAX_COMPILED_FORMS), Any[], Ref(false))
end

"""
    assemble_form!(b, a::FormAssembler, L, key) -> b

Writes into `b` the vector of the linear form `L` (`v -> ∫(…)dΩ`) on the test space of the assembler.
"""
function assemble_form!(b, a::LazyAssembler, L, key)
  assemble_vector!(L, b, a.assem, a.V)
  b
end

_as_residual(L) = (u, v) -> L(v)
_zero_tangent(dΩ) = (u, du, v) -> ∫(0.0 * (du ⊙ v))dΩ      # ⊙: scalar and vector-valued fields

function assemble_form!(b, a::CompiledAssembler, L, key)
  res, jac = _as_residual(L), _zero_tangent(a.dΩ)
  i = findfirst(k -> isequal(k, key), a.keys)
  if i !== nothing
    op = update(a.ops[i], res, jac)
    # most recently used last
    deleteat!(a.keys, i); deleteat!(a.ops, i)
    push!(a.keys, key); push!(a.ops, op)
  else
    # a new key: derived from the last operator, so tables, geometry and colouring are shared
    op = isempty(a.ops) ? KernelOperator(res, jac, a.U, a.V; jacobian = :matrixfree) : update(last(a.ops), res, jac)
    j = findfirst(k -> isequal(k, key), a.evicted)
    if j !== nothing                      # dropped before and needed again: one more slot
      deleteat!(a.evicted, j)
      if a.capacity[] < MAX_CAPACITY
        a.capacity[] += 1
      elseif !a.warned[]
        a.warned[] = true
        @warn "HyperShape: more than $MAX_CAPACITY forms assembled on one space; some are compiled again at every use"
      end
    end
    if length(a.ops) >= a.capacity[]
      k = findfirst(k -> !isequal(k, :forward), a.keys)   # least recently used, never the filter's load vector
      k === nothing && (k = 1)
      push!(a.evicted, a.keys[k])
      length(a.evicted) > MAX_CAPACITY && popfirst!(a.evicted)
      deleteat!(a.keys, k); deleteat!(a.ops, k)
    end
    push!(a.keys, key)
    push!(a.ops, op)
  end
  residual!(b, op, a.x0)
  b
end
