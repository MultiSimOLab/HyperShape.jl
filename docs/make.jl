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
        assets = ["assets/favicon.ico"],
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Design" => "design.md",
        "Validation" => "validation.md",
        "Benchmarks" => "benchmarks.md",
        "Roadmap" => "roadmap.md",
        "API reference" => ["api.md", "api/methods.md", "api/functionals.md", "api/variables.md", "api/operators.md",
                            "api/optimizers.md", "api/checks.md", "api/postprocess.md"],
    ],
    checkdocs = :exports,
)

deploydocs(;
    repo = "github.com/MultiSimOLab/HyperShape.jl",
    devbranch = "main",
)
