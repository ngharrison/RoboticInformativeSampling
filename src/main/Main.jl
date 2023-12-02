# allows using modules defined in any file in project src directory
mod_dir = dirname(Base.active_project()) * "/src/modules"
if mod_dir ∉ LOAD_PATH
    push!(LOAD_PATH, mod_dir)
end

# set the logging level: Info or Debug
using Logging
global_logger(ConsoleLogger(stderr, Logging.Info))

using Missions: simMission, ausMission, nswMission, conradMission, rosMission
using BeliefModels: outputCorMat
using Visualization: visualize
using Metrics: calcMetrics
using Outputs: save

## initialize data for mission
mission = simMission(num_samples=10)

## run search alg
@time samples, beliefs = mission(visuals=false, sleep_time=0.0);
@debug "output determination matrix:" outputCorMat(beliefs[end]).^2
# save(mission, samples, beliefs; animation=true)

## calculate errors
metrics = calcMetrics(mission, beliefs, 1)

## save outputs
save(metrics)
