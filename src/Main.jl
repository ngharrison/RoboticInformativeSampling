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

## initialize data for mission
mission = simMission()

## run search alg
@time samples, beliefModel = mission(visuals=true, sleep_time=0.0);

println()
println("Output correlations:")
display(outputCorMat(beliefModel))
