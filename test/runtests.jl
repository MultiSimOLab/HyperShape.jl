using HyperShape
using Test

@testset "HyperShape.jl" begin

    @testset "Package loads and exports its public names" begin
        legacy = (:FEFunctional, :OptimFEVariable, :update!, :pushforward!, :pullback!,
                  :adimensionalize!, :evaluate_derivative!, :evaluate_objective, :evaluate!,
                  :get_space, :get_derivative)
        @test !isdefined(HyperShape, :TopOpt)   # earlier prototype, removed
        for name in legacy
            @test !(name in names(HyperShape))
        end
        for name in (:OC, :SiMPL, :Optimization, :SIMP, :StateOperator, :Functional)
            @test name in names(HyperShape)
        end
    end

    include("common.jl")
    include("Optimizers/runtests.jl")
    include("Variables/runtests.jl")
    include("Functionals/runtests.jl")
    include("Checks/runtests.jl")
    include("Methods/runtests.jl")

end
