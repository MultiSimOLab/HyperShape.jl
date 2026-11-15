"""
    HyperShape.Variables

Design variables at three linked levels:

| Level | Object | What it is |
|---|---|---|
| z | [`OptVariables`](@ref) | what the optimiser sees: one vector made of blocks, with bounds |
| x | [`DesignFields`](@ref) | design fields on the design mesh (cell-wise P0 or nodal P1) |
| θ | [`PhysicalFields`](@ref) | physical fields read by the weak forms at the quadrature points |

Each level is built from the previous one, and only HyperShape writes them (in place). The maps x → θ
([`HelmholtzFilter`](@ref), [`Heaviside`](@ref), [`Identity`](@ref)) are composed with `∘` in the mathematical
order: `Heaviside(β = 1.0, η = 0.5) ∘ HelmholtzFilter(Sf; measure = dΩ)` filters first and projects after.

Sensitivities travel backwards with [`design_gradient!`](@ref): a linear form in δθ for each physical field gives the
derivative with respect to z.

The vectors HyperShape assembles (load vector of a filter, sensitivities) use `backend = :compiled` by default
(HyperFEMKernels: forms compiled once into threaded kernels) or `backend = :lazy` (Gridap's assembler); see
[`HelmholtzFilter`](@ref) and [`PhysicalFields`](@ref). Both give the same vectors up to rounding. The compiled
backend is faster from tens of thousands of cells on, already with one thread, and scales with the threads
(`julia -t N`); on meshes of about a thousand cells both take well under a millisecond per vector.
"""
module Variables

using LinearAlgebra
using Gridap
using Gridap.FESpaces
using Gridap.Algebra: residual!
using HyperFEM: get_spaces
using HyperFEMKernels: KernelOperator, update
using ..Operators: LinearOperator, solve_linear!
import ..Operators: version, refresh!

export FieldVariable, OptVariables, DesignFields, PhysicalFields
export Identity, HelmholtzFilter, Heaviside
export single_field_design

include("OptVariables.jl")
include("DesignFields.jl")
include("Assembly.jl")
include("Maps.jl")
include("PhysicalFields.jl")

refresh!(x::DesignFields, ctx) = update_fields!(x, ctx)
refresh!(θ::PhysicalFields, ctx) = update_fields!(θ, ctx)

"""
    single_field_design(name, V; bounds, init, physical, backend = :compiled) -> (z, x, θ)

Shortcut for one design field and its physical fields with the same name:

```julia
z, x, θ = single_field_design(:ρ, Vρ; bounds = (0.0, 1.0), init = 0.5,
                              physical = Heaviside(β = 1.0, η = 0.5) ∘ HelmholtzFilter(Sf; measure = dΩ))
```

is `OptVariables(ρ = FieldVariable(Vρ; bounds, init))`, `DesignFields(z; ρ = Identity())` and
`PhysicalFields(x; ρ = physical, backend)`.
"""
function single_field_design(name::Symbol, V; bounds, init, physical, backend::Symbol = :compiled)
  z = OptVariables(; NamedTuple{(name,)}((FieldVariable(V; bounds, init),))...)
  x = DesignFields(z; NamedTuple{(name,)}((Identity(),))...)
  θ = PhysicalFields(x; backend, NamedTuple{(name,)}((physical,))...)
  z, x, θ
end

end
