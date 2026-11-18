# Adjoint of a forward state: what the user declares (`AdjointProblem`, `SameMatrix`, `SelfAdjoint`) and the cache that
# HyperShape builds once inside the StateOperator (`AdjointCache`). The reverse sweep that uses it (seeds from the
# functionals, adjoint solve, forms in δθ) lives in `HyperShape.Functionals`.

"""
    SameMatrix()

Operator of an [`AdjointProblem`](@ref) for a forward state over a linear and symmetric `StaticLinearModel`: the adjoint is
solved with the matrix and the factorisation of the forward model itself (no assembly and no factorisation for the
adjoint, one back substitution per gradient). The forward model must be re-assembled by its own solve function (the
default one does), because the adjoint solve overwrites its right-hand side vector.
"""
struct SameMatrix end

"""
    SelfAdjoint()

Value of `dJdu` in a [`Functional`](@ref HyperShape.Functionals.Functional) whose derivative with respect to the state
is the load of the state (`∂J/∂u = ℓ`, for example the compliance `J = ℓ(u)` of a linear and symmetric problem): the
adjoint is the state itself, `p = u`, and no adjoint is solved. Valid only when the Dirichlet values of the state are
zero (checked).
"""
struct SelfAdjoint end

"""
    AdjointProblem(op; sensitivity, solve = <default>, depends = (), backend = :compiled)

Adjoint problem of a forward [`StateOperator`](@ref), given with its `adjoint` keyword. With `R(u; θ)[v]` the residual of
the forward problem and `K = ∂R/∂u` its tangent, the adjoint `p` solves `Kᵀ p = Σ ∂J/∂u` (the seeds of the
functionals that read the state, written by them in `dJdu`) and the state contributes `− pᵀ ∂R/∂θ` to the derivative
with respect to the physical fields. HyperShape puts the signs; the user writes the partial derivatives.

- `op`: who solves `Kᵀ p = b`. Either a HyperFEM model that carries the transposed tangent alone and the user's
  solver, usually `StaticLinearModel((p, w) -> jac(uh, w, p), U0, V0, bc0; ls)` (test and trial swapped, evaluated at
  the forward state, on a space with homogeneous Dirichlet values), or [`SameMatrix`](@ref)`()` for a linear and
  symmetric forward state (reuses its factorisation).
- `sensitivity`: `p -> (δ -> ∫(…)dΩ)`, the linear form `pᵀ ∂R/∂θ [δ]` (without sign), or a named tuple of such forms
  (`p -> (ρ = δ -> …, E = δ -> …)`) when θ has several fields. Written once per state, it serves every functional
  that reads the state.
- `solve` (only with a model): function `(model, b, ctx, stale) -> x` of a [`LinearOperator`](@ref); default
  `solve!(m, b; Assembly = stale)` for a `StaticLinearModel` ([`default_solve`](@ref)). The matrix is re-assembled
  (`stale = true`) only when the forward state changed, so the factorisation serves every gradient of a design.
- `depends`: objects that the adjoint matrix reads besides those of the forward state (a [`Parameter`](@ref), …).
- `backend`: who assembles the seeds `∂J/∂u` on the test space of the adjoint: `:compiled` (HyperFEMKernels) or
  `:lazy` (Gridap).

Cost per gradient of a design: one adjoint solve (a back substitution after the first one) for all the functionals
that read the state, none when they are all [`SelfAdjoint`](@ref).
"""
struct AdjointProblem{O,SE,F,D}
  op::O
  sensitivity::SE
  solve::F
  depends::D
  backend::Symbol
end

function AdjointProblem(op; sensitivity, solve = nothing, depends = (), backend::Symbol = :compiled)
  _check_backend(backend)
  op isa SameMatrix && solve !== nothing &&
    throw(ArgumentError("AdjointProblem: SameMatrix() solves with the forward model; solve is only for an adjoint model"))
  (op isa SameMatrix || applicable(HyperFEM.get_spaces, op)) ||
    throw(ArgumentError("AdjointProblem: op must be a HyperFEM model with the transposed tangent (a StaticLinearModel) or SameMatrix(); got $(typeof(op))"))
  AdjointProblem(op, sensitivity, solve, _check_depends(_as_depends(depends)), backend)
end

Base.show(io::IO, a::AdjointProblem) =
  print(io, "AdjointProblem(", a.op isa SameMatrix ? "SameMatrix()" : nameof(typeof(a.op)), ")")

"""
    AdjointCache

What HyperShape builds from an [`AdjointProblem`](@ref) when the [`StateOperator`](@ref) is constructed: the operator that
solves `Kᵀ p = b`, the adjoint field `p` (a fixed object updated in place), the accumulated right-hand side and the
assembler of the seeds. Internal.
"""
mutable struct AdjointCache{O,P,A,SE}
  op::O                       # LinearOperator over the adjoint model, or SameMatrix()
  p::P                        # adjoint field, values updated in place
  rhs::Vector{Float64}        # accumulated seeds Σ wᵢ ∂Jᵢ/∂u
  tmp::Vector{Float64}
  assem::A                    # assembler of the seeds on the test space of the adjoint
  sensitivity::SE
  has_rhs::Bool               # a seed other than a self-adjoint one was added
  self_weight::Float64        # Σ of the weights of the self-adjoint seeds (p = self_weight · u + K⁻ᵀ rhs)
  nsolves::Int                # adjoint solves done
end

_adjoint_cache(::Nothing, model, deps) = nothing

function _adjoint_cache(a::AdjointProblem, model, deps)
  if a.op isa SameMatrix
    model isa HyperFEM.StaticLinearModel && !(model isa HyperFEM.StaticLinearModel{<:AbstractVector}) ||
      throw(ArgumentError("AdjointProblem: SameMatrix() needs a forward StaticLinearModel with one right-hand side; got a $(nameof(typeof(model)))"))
    U, V = HyperFEM.get_spaces(model)
    n = num_free_dofs(V)
    p = FEFunction(V, zeros(n))            # the test space: homogeneous Dirichlet values
    op = a.op
  else
    _check_homogeneous_trial(a.op, "AdjointProblem")
    U, V = HyperFEM.get_spaces(a.op)
    n = num_free_dofs(V)
    num_free_dofs(U) == n ||
      throw(DimensionMismatch("AdjointProblem: the adjoint model has $(num_free_dofs(U)) unknowns and $n equations"))
    p = FEFunction(U, zeros(n))
    solve = a.solve === nothing ? default_solve(LinearOperator, a.op) : a.solve
    op = LinearOperator(a.op, _unique_deps((deps..., a.depends...)); solve)
    invalidate!(op)          # the matrix of the adjoint model was assembled before the state was solved
  end
  AdjointCache(op, p, zeros(n), zeros(n), form_assembler(a.backend, U, V), a.sensitivity, false, 0.0, 0)
end

"""
    has_adjoint(S::StateOperator) -> Bool

Whether `S` was built with an [`AdjointProblem`](@ref), so that functionals can be differentiated through it.
"""
has_adjoint(S) = false

"""
    reset_adjoint!(S::StateOperator)

Starts a reverse sweep: the accumulated seeds of the adjoint of `S` are set to zero.
"""
function reset_adjoint!(S)
  c = S.adjoint
  fill!(c.rhs, 0.0)
  c.has_rhs = false
  c.self_weight = 0.0
  S
end

"""
    is_seeded(S::StateOperator) -> Bool

Whether a functional has added a seed to the adjoint of `S` since the last [`reset_adjoint!`](@ref) or solve.
"""
is_seeded(S) = S.adjoint.has_rhs || S.adjoint.self_weight != 0

"""
    add_seed!(S::StateOperator, form, w, key)

Adds `w ⋅ form` to the right-hand side of the adjoint of `S`; `form` is the linear form `w -> ∫(…)` of `∂J/∂u`, assembled
on the test space of the adjoint (`key` identifies it for the compiled assembler).
"""
function add_seed!(S, form, w::Float64, key)
  c = S.adjoint
  assemble_form!(c.tmp, c.assem, form, key)
  c.rhs .+= w .* c.tmp
  c.has_rhs = true
  S
end

"""
    add_self_seed!(S::StateOperator, w)

Adds `w ⋅ ∂J/∂u` for a [`SelfAdjoint`](@ref) functional: its contribution to the adjoint is `w ⋅ u`, without a solve.
"""
function add_self_seed!(S, w::Float64)
  S.adjoint.self_weight += w
  S
end

_solve_adjoint!(pv, op::LinearOperator, S, rhs, ctx) = solve_linear!(pv, op, rhs, ctx)
function _solve_adjoint!(pv, ::SameMatrix, S, rhs, ctx)
  copyto!(pv, rhs)
  S.model(pv; Assembly = false)            # the factorisation of the forward model; writes the solution into pv
  pv
end

"""
    solve_adjoint!(S::StateOperator, ctx) -> p

Solves the adjoint of `S` with the accumulated seeds (one solve, none if they are all self-adjoint) and returns the
adjoint field `p`. The seeds are consumed: [`is_seeded`](@ref) is `false` afterwards. `S` must have converged.
"""
function solve_adjoint!(S, ctx)
  c = S.adjoint
  pv = get_free_dof_values(c.p)
  u = get_free_dof_values(S.state)
  if c.has_rhs
    _solve_adjoint!(pv, c.op, S, c.rhs, ctx)
    c.nsolves += 1
    c.self_weight == 0 || (pv .+= c.self_weight .* u)
  else
    pv .= c.self_weight .* u
  end
  c.has_rhs = false
  c.self_weight = 0.0
  c.p
end

"""
    adjoint_sensitivity(S::StateOperator, p)

The forms `pᵀ ∂R/∂θ` declared in the `sensitivity` of the [`AdjointProblem`](@ref) of `S`, at the adjoint `p` (without sign).
"""
adjoint_sensitivity(S, p) = S.adjoint.sensitivity(p)

num_solves(c::AdjointCache) = c.nsolves
invalidate!(c::AdjointCache) = (c.op isa LinearOperator && invalidate!(c.op); c)
