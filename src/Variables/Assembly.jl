# Assembly of the vectors of linear forms (load vector of a filter, sensitivities), by one of two backends:
#   :compiled  HyperFEMKernels: the form is compiled once into threaded kernels and re-evaluated at every call
#   :lazy      Gridap's assembler
#
# A compiled assembler keeps one kernel operator per `key`. Forms assembled under the same key must have the same
# structure (same measures, operations and fields, with new values): then only the values are rebound. A form with
# another structure under the same key is compiled again at every call, which is correct but slow.

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
  ops::Dict{Any,Any}       # key => KernelOperator
end

const BACKENDS = (:compiled, :lazy)

function _check_backend(backend)
  backend in BACKENDS || throw(ArgumentError("backend must be :compiled or :lazy, got $(repr(backend))"))
  backend
end

function form_assembler(backend::Symbol, U, V)
  _check_backend(backend)
  backend === :lazy && return LazyAssembler(SparseMatrixAssembler(U, V), V)
  CompiledAssembler(U, V, Measure(get_triangulation(V), 1), zeros(num_free_dofs(U)), Dict{Any,Any}())
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
  op = get(a.ops, key, nothing)
  op = try
    op === nothing ? KernelOperator(res, jac, a.U, a.V; jacobian = :matrixfree) : update(op, res, jac)
  catch e
    e isa InterruptException && rethrow()
    throw(ErrorException("HyperShape: the compiled backend (HyperFEMKernels) could not compile a linear form; " *
                         "use backend = :lazy for this object. Cause: " * sprint(showerror, e)))
  end
  a.ops[key] = op
  residual!(b, op, a.x0)
  b
end
