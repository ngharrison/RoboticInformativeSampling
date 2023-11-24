# allows using modules defined in any file in project src directory
mod_dir = dirname(Base.active_project()) * "/src/modules"
if mod_dir ∉ LOAD_PATH
    push!(LOAD_PATH, mod_dir)
end

using Documenter
using Maps, Missions, BeliefModels,
      Samples, SampleCosts, Paths, Visualization

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
