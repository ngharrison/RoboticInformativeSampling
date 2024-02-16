# allows using modules defined in any file in project src directory
mod_dir = dirname(Base.active_project()) * "/src/modules"
if mod_dir ∉ LOAD_PATH
    push!(LOAD_PATH, mod_dir)
end

using Missions: Mission
using BeliefModels: BeliefModel
using Samples: Sample
using Outputs: output_dir, output_ext
using ROSInterface: ROSConnection

using Statistics: mean, std
using FileIO: load
using Plots
using Plots: mm

file_name = output_dir * "2024-02-15-12-02-46_mission.jld2"
file_name = output_dir * "2024-02-15-12-22-16_mission.jld2"
file_name = output_dir * "2024-02-15-12-56-06_mission.jld2"

data = load(file_name)
mission = data["mission"]
beliefs = data["beliefs"]
samples = data["samples"]

using Visualization
quantities = eachindex(mission.sampler)
sampleCost = mission.sampleCostType(mission, samples, beliefs[end], quantities)
display(visualize(mission, samples, beliefs[end], mission.occupancy.lb; quantity=1))
