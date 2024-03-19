#!/usr/bin/env julia

using Pkg
Pkg.activate(Base.source_dir() * "/../..")

# allows using modules defined in any file in project modules directory
mod_dir = dirname(Base.active_project()) * "/src/modules"
if mod_dir ∉ LOAD_PATH
    push!(LOAD_PATH, mod_dir)
end

# set the logging level: Info or Debug
using Logging
global_logger(ConsoleLogger(stderr, Logging.Debug))

using Visualization: vis
using Outputs: save

include("../missions/pye_farm.jl")

## initialize data for mission
mission = pyeFarmMission()

## run search alg
@time samples, beliefs = mission(
    sleep_time=0.0
) do M, samples, beliefModel, sampleCost, new_loc
    vis(beliefModel, samples, new_loc, M.occupancy, 1)
    save(M, samples, beliefModel)
end;
save(mission, samples, beliefs)

## save outputs
# saveBeliefMapToPng(beliefs[end], mission.occupancy)

