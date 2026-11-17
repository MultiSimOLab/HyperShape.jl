using HyperShape
using Test

@testset "HyperShape.jl" begin

    @testset "Package loads and exports its public names" begin
        for name in (:OC, :SiMPL, :Optimization, :DensityMethod, :StateOperator, :Functional, :solve!)
            @test name in names(HyperShape)
        end
        for name in (:FieldWriter, :write_fields)          # public, not exported (to move to HyperFEM)
            @test Base.ispublic(HyperShape, name) && !Base.isexported(HyperShape, name)
        end
    end

    include("common.jl")
    include("Optimizers/runtests.jl")
    include("Variables/runtests.jl")
    include("Functionals/runtests.jl")
    include("Checks/runtests.jl")
    include("Methods/runtests.jl")
    include("PostProcess/runtests.jl")
    include("Validation/runtests.jl")
    include("Examples/runtests.jl")

end
