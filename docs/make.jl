using Documenter
using HyperShape

makedocs(;
    modules = [HyperShape, HyperShape.Optimizers, HyperShape.Operators, HyperShape.Variables, HyperShape.Functionals, HyperShape.Checks, HyperShape.PostProcess, HyperShape.Methods],
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
        "Getting started" => "getting_started.md",
        "Design" => "design.md",
        "Validation" => "validation.md",
        "Roadmap" => "roadmap.md",
        "API reference" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(;
    repo = "github.com/MultiSimOLab/HyperShape.jl",
    devbranch = "main",
)
