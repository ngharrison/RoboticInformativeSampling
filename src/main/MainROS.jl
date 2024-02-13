#!/usr/bin/env julia

using Pkg
Pkg.activate(Base.source_dir() * "/../..")

# allows using modules defined in any file in project modules directory
mod_dir = dirname(Base.active_project()) * "/src/modules"
if mod_dir ∉ LOAD_PATH
    push!(LOAD_PATH, mod_dir)
end

using Outputs: save

# set the logging level: Info or Debug
using Logging
global_logger(ConsoleLogger(stderr, Logging.Info))

using Missions: rosMission

## initialize data for mission
mission = rosMission(num_samples=4)

## run search alg
@time samples, beliefs = mission(visuals=false, sleep_time=0.0);
save(mission, samples, beliefs)

## save outputs
# saveBeliefMapToPng(beliefs[end], mission.occupancy)
