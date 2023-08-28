# allows using modules defined in any file in project src directory
src_dir = dirname(Base.active_project()) * "/src"
if src_dir ∉ LOAD_PATH
    push!(LOAD_PATH, src_dir)
end

# set the logging level: Info or Debug
using Logging
global_logger(ConsoleLogger(stderr, Logging.Info))

using Missions: simMission, ausMission, conradMission, rosMission
using BeliefModels: outputCorMat
using Visualization: visualize
using Metrics: calcMetrics
using Outputs: save

## initialize data for mission
mission = simMission()

## run search alg
@time samples, beliefs = mission(visuals=true, sleep_time=0.5);
@debug "output determination matrix:" outputCorMat(beliefs[end]).^2
# save(mission, samples, beliefs; animation=true)

## calculate errors
metrics = calcMetrics(mission, beliefs, 1)

## save outputs
save(metrics)
