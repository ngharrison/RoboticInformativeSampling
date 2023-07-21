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
using Outputs: saveOutputs

## initialize data for mission
mission = ausMission()

## run search alg
@time samples, beliefs = mission(visuals=true, sleep_time=0.0);

@debug "output correlation matrix:" outputCorMat(beliefs[end])

## calculate errors
metrics = calcMetrics(mission, samples, beliefs)

## save outputs
# saveOutputs(mission, samples, beliefs, metrics; save_animation=true)
