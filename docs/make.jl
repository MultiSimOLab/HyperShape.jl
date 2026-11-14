using Documenter
using HyperShape

makedocs(;
    modules = [HyperShape, HyperShape.TopOpt, HyperShape.Optimizers, HyperShape.Operators, HyperShape.Variables, HyperShape.Functionals],
    authors = "MultiSimOLab",
    sitename = "HyperShape.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://MultiSimOLab.github.io/HyperShape.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Design" => "design.md",
        "Roadmap" => "roadmap.md",
        "API reference" => "api.md",
    ],
    checkdocs = :none,
)

deploydocs(;
    repo = "github.com/MultiSimOLab/HyperShape.jl",
    devbranch = "main",
)
