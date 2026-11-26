using HyperShape
using Test

# The suite in groups, so that CI runs them as parallel jobs: HYPERSHAPE_TEST_GROUP = core | methods | rd | validation
# (default: all of them)
const GROUP = get(ENV, "HYPERSHAPE_TEST_GROUP", "all")
const GROUPS = Dict("core" => ("Optimizers", "Variables", "Functionals", "Checks", "PostProcess"),
                    "methods" => ("Methods",),
                    "rd" => ("ReactionDiffusion",),
                    "validation" => ("Validation", "Examples"))
GROUP == "all" || haskey(GROUPS, GROUP) ||
  error("HYPERSHAPE_TEST_GROUP must be one of all, $(join(sort!(collect(keys(GROUPS))), ", ")); got $GROUP")
const DIRS = GROUP == "all" ? (GROUPS["core"]..., GROUPS["methods"]..., GROUPS["rd"]..., GROUPS["validation"]...) : GROUPS[GROUP]

@testset "HyperShape.jl" begin

    if GROUP in ("all", "core")
        @testset "Package loads and exports its public names" begin
            for name in (:OC, :SiMPL, :Optimization, :DensityMethod, :ReactionDiffusion, :ExactVolume, :L1Mean, :L2Norm,
                         :MaxAbs, :DoubleWell, :StateOperator, :Functional, :solve!, :hole_seed)
                @test name in names(HyperShape)
            end
            for name in (:FieldWriter, :write_fields)          # public, not exported (to move to HyperFEM)
                @test Base.ispublic(HyperShape, name) && !Base.isexported(HyperShape, name)
            end
        end
    end

    include("common.jl")
    for d in DIRS
        include(joinpath(d, "runtests.jl"))
    end

end
