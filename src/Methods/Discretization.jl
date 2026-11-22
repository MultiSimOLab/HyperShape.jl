# The discretisation of a run: whether and when solve! builds the problem again (rebuild events).

"""
    Discretization

Supertype of the discretisations given to `solve!(…; discretization)`: they say whether the problem is built again
during the run, when, and how. [`FixedMesh`](@ref)`()` (the default) never does; `HyperShape.Methods.RebuildCheck`
does it on the same mesh; adaptive meshes come later. The contract (public, not exported):

- `needs_setup(d)`: whether the problem must have factories (`Optimization(model; design_setup, physics_setup)`);
- `due(d, k, info, history, ctx) -> Bool`: after the accepted iteration `k` (not the last one, nor the iteration a
  run resumes from), whether to rebuild; `history` is the live [`OptHistory`](@ref) (its first `history.n` entries
  are filled);
- `rediscretize(d, problem, state::OptState, ctx) -> Optimization`: the new problem (for example
  `HyperShape.Methods.rebuild(problem)`, or `rebuild(problem; what = :physics)` to keep the design), called with
  `ctx.phase == :setup`; `solve!` checks it (new variables, unless the design is kept, and new functionals; the same
  blocks and number of values of z, the same frozen cells, the same constraints with the same bounds and kind, the
  same monitors and output, a continued `TanhProjection` in the new maps) and continues from `state`. A change of
  mesh (another number of values) is not supported yet;
- `discretization_label(d)`: the line of the banner.

A rebuild costs the calls of the factories, one forward solve and one filter (counted in the summary of the run, not
in the rows of the history). An error at a rebuild, or before the first iteration after it, keeps the state of the
rebuild (with the memory of the optimiser) in `problem.state[]` and in the checkpoint.
"""
abstract type Discretization end

"""
    FixedMesh()

The default discretisation of [`solve!`](@ref): the mesh and the problem do not change during the run. Any problem
(built from its functionals or with factories).
"""
struct FixedMesh <: Discretization end

"""
    HyperShape.Methods.RebuildCheck(; at = nothing, every = nothing, what = :design)

Builds the problem again on the same mesh, calling the factories of `Optimization(model; design_setup,
physics_setup)` after the accepted iterations `at` (an integer or a collection) or every `every` iterations: both
factories (`what = :design`) or only `physics_setup`, with the same design (`what = :physics`). The run continues from
the state captured at that iteration (the path of `resume`), so with linear states solved by direct solvers it is the
run of `FixedMesh()` bit for bit (the counts of solves aside: one forward solve and one filter more per rebuild, in
the summary). A run resumed from a checkpoint written at an iteration of a rebuild does not rebuild there again. It checks that the factories capture nothing of the problem they replace, before paying for a change of mesh:
at each rebuild the objective at the rebuilt problem is compared with the one of the history (`identical`, or its
relative difference, on screen). A nonlinear state or an iterative solver starts from another initial guess after a
rebuild, and agrees only to its tolerance.
"""
struct RebuildCheck <: Discretization
  at::Vector{Int}
  every::Int
  what::Symbol
end

function RebuildCheck(; at = nothing, every = nothing, what::Symbol = :design)
  (at === nothing) == (every === nothing) && throw(ArgumentError("RebuildCheck: give exactly one of at (iterations) and every"))
  what in (:design, :physics) ||
    throw(ArgumentError("RebuildCheck: what must be :design (both factories) or :physics (physics_setup only)"))
  if every !== nothing
    (every isa Integer && every >= 1) || throw(ArgumentError("RebuildCheck: every must be a positive integer"))
    return RebuildCheck(Int[], Int(every), what)
  end
  ks = sort!(unique!(collect(Int, at isa Integer ? (at,) : at)))
  (isempty(ks) || first(ks) < 1) && throw(ArgumentError("RebuildCheck: at must be positive iterations"))
  RebuildCheck(ks, 0, what)
end

Base.show(io::IO, d::RebuildCheck) =
  print(io, "RebuildCheck(", d.every > 0 ? "every = $(d.every)" : "at = $(d.at)", ", what = :", d.what, ")")

needs_setup(::Discretization) = true
needs_setup(::FixedMesh) = false

due(::FixedMesh, k, info, history, ctx) = false
due(d::RebuildCheck, k, info, history, ctx) = d.every > 0 ? k % d.every == 0 : insorted(k, d.at)

function rediscretize end
rediscretize(d::RebuildCheck, problem, state, ctx) = rebuild(problem; what = d.what)
rediscretize(d::FixedMesh, problem, state, ctx) = throw(ArgumentError("rediscretize: FixedMesh does not rebuild"))

discretization_label(d::Discretization) = string(nameof(typeof(d)))
discretization_label(::FixedMesh) = "fixed mesh"
discretization_label(d::RebuildCheck) =
  "RebuildCheck, " * (d.what === :design ? "design and physics" : "physics") * " rebuilt " *
  (d.every > 0 ? "every $(d.every) iterations" : "at iteration" * (length(d.at) > 1 ? "s " : " ") * join(d.at, ", ")) *
  " (same mesh)"
