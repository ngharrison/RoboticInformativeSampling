# script for analyzing data from missions
# can be run after Main.jl or by opening a saved file

# allows using modules defined in any file in project src directory
src_dir = dirname(Base.active_project()) * "/src"
if src_dir ∉ LOAD_PATH
    push!(LOAD_PATH, src_dir)
end

using Missions: Mission
using BeliefModels: BeliefModel
using Samples: Sample
using FileIO: load
using Plots: plot

data = load("../output/2023-07-21-17-56-40.jdl2")

mission = data["mission"]
beliefs = data["beliefs"]
samples = data["samples"]
metrics = data["metrics"]

plot(
    plot(1:mission.num_samples, metrics.mae),
    plot(1:mission.num_samples, metrics.mu),
    plot(1:mission.num_samples, metrics.mb),
    layout = (3, 1)
)
