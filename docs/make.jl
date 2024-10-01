
using Documenter

using InformativeSampling: Maps, Missions, BeliefModels, Kernels,
      Samples, SampleCosts, Paths, ROSInterface

using InformativeSamplingUtils: DataIO, Visualization, Metrics

makedocs(
    sitename="InformativeSampling.jl",
    remotes=nothing,
    pages = [
        "index.md",
        "application.md",
        "core.md",
        "utilities.md",
        "code_patterns.md",
        "julia_tips.md",
    ],
    format = Documenter.HTML(prettyurls=false)
)
