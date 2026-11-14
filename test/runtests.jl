using HyperShape
using Test

@testset "HyperShape.jl" begin

    @testset "Package loads and exports its public names" begin
        names = (:FEFunctional, :OptimFEVariable, :update!, :pushforward!, :pullback!,
                 :adimensionalize!, :evaluate_derivative!, :evaluate_objective, :evaluate!,
                 :get_space, :get_derivative)
        for name in names
            @test isdefined(HyperShape, name)
        end
    end

end
