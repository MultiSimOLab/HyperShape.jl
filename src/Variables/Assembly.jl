# Assembly of the vectors of linear forms (load vector of a filter, sensitivities), by one of two backends:
#   :compiled  HyperFEMKernels: the form is compiled once into threaded kernels and re-evaluated at every call
#   :lazy      Gridap's assembler
#
# A compiled assembler keeps the kernel operator of the last forms it assembled, one per `key` (at most
# `MAX_COMPILED_FORMS`, the oldest is dropped). All of them share tables, geometry and colouring: the operator of a
# new key is derived from an existing one. A form assembled again under its key with the same structure (same
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
  keys::Vector{Any}        # keys and their kernel operators, oldest first
  ops::Vector{Any}
end

const MAX_COMPILED_FORMS = 16

const BACKENDS = (:compiled, :lazy)

function _check_backend(backend)
  backend in BACKENDS || throw(ArgumentError("backend must be :compiled or :lazy, got $(repr(backend))"))
  backend
end

function form_assembler(backend::Symbol, U, V)
  _check_backend(backend)
  backend === :lazy && return LazyAssembler(SparseMatrixAssembler(U, V), V)
  CompiledAssembler(U, V, Measure(get_triangulation(V), 1), zeros(num_free_dofs(U)), Any[], Any[])
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
_zero_tangent(dΩ) = (u, du, v) -> ∫(0.0 * du * v)dΩ

function assemble_form!(b, a::CompiledAssembler, L, key)
  res, jac = _as_residual(L), _zero_tangent(a.dΩ)
  i = findfirst(k -> isequal(k, key), a.keys)
  if i !== nothing
    op = update(a.ops[i], res, jac)
    a.ops[i] = op
  else
    # a new key: derived from the last operator, so tables, geometry and colouring are shared
    op = isempty(a.ops) ? KernelOperator(res, jac, a.U, a.V; jacobian = :matrixfree) : update(last(a.ops), res, jac)
    if length(a.ops) >= MAX_COMPILED_FORMS
      popfirst!(a.keys)
      popfirst!(a.ops)
    end
    push!(a.keys, key)
    push!(a.ops, op)
  end
  residual!(b, op, a.x0)
  b
end
