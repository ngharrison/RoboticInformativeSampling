using Documenter

using AdaptiveSampling
      Samples, SampleCosts, Paths, Visualization, ROSInterface

makedocs(
    sitename="AdaptiveSampling.jl",
    remotes=nothing,
    pages = [
        "index.md",
        "code_parts.md",
        "code_patterns.md",
        "julia_tips.md",
    ]
)
