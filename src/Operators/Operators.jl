"""
    HyperShape.Operators

Operators over HyperFEM computational models.

- [`LinearOperator`](@ref): auxiliary linear solves (filter, reaction–diffusion step, velocity extension). The model
  carries the bilinear form only and HyperShape supplies the right-hand side.
- `StateOperator` (forward and adjoint states): planned.

Both declare their dependencies explicitly (`depends = ...`): objects with a [`version`](@ref) counter.
"""
module Operators

export LinearOperator

"""
    version(obj) -> Int

Counter that grows every time `obj` changes. Objects listed in the `depends` of an operator implement it.
"""
function version end

"""
    LinearOperator(model; depends, solve)

Linear solve with a matrix built by a HyperFEM model (typically a `StaticLinearModel` with the bilinear form alone)
and a right-hand side supplied by HyperShape.

- `model`: the computational model; its matrix and solver are the user's.
- `depends`: tuple of objects the matrix depends on (each implements [`version`](@ref)); `()` if the matrix never
  changes. Mandatory.
- `solve`: user function `(model, b, ctx, stale) -> x` that solves with the right-hand side `b` and returns the
  solution vector. `stale` is `true` when something in `depends` changed since the last solve, so the user can
  re-assemble, for example `solve = (m, b, ctx, stale) -> solve!(m, b; Assembly = stale)`. Mandatory.

The matrix must be the one the model assembled at construction (no dependency has changed yet).
"""
mutable struct LinearOperator{M,F,D}
  model::M
  solve::F
  depends::D
  seen::Vector{Int}
end

function LinearOperator(model; depends, solve)
  depends isa Tuple || throw(ArgumentError("depends must be a tuple (use () for a matrix that never changes)"))
  for d in depends
    applicable(version, d) || throw(ArgumentError("depends: $(typeof(d)) does not implement HyperShape.Operators.version"))
  end
  LinearOperator(model, solve, depends, Int[version(d) for d in depends])
end

get_model(op::LinearOperator) = op.model

"""
    solve_linear!(x, op::LinearOperator, b, ctx) -> x

Solves with the right-hand side `b` through the user's `solve` and copies the solution into `x`.
"""
function solve_linear!(x, op::LinearOperator, b, ctx)
  stale = false
  for (i, d) in enumerate(op.depends)
    v = version(d)
    if v != op.seen[i]
      stale = true
      op.seen[i] = v
    end
  end
  y = op.solve(op.model, b, ctx, stale)
  length(y) == length(x) ||
    throw(DimensionMismatch("the solve function of the LinearOperator returned $(length(y)) values, expected $(length(x))"))
  copyto!(x, y)
end

end
