
using Documenter

using InformativeSampling: Maps, Missions, BeliefModels, Kernels,
      Samples, SampleCosts, Paths, ROSInterface

makedocs(
    sitename="InformativeSampling.jl",
    remotes=nothing,
    pages = [
        "index.md",
        "package.md",
        "code_patterns.md",
        "application.md",
        "julia_tips.md",
    ],
    format = Documenter.HTML(prettyurls=false)
)
