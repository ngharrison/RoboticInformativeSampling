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

mission_peaks = [1,2]
num_runs = 2
metrics = Array{Any, 2}(undef, (length(mission_peaks), num_runs))

for (i, num_peaks) in enumerate(mission_peaks)
    ## initialize data for mission
    mission = simMission(; num_peaks, seed_val=i)
    for j in 1:num_runs
        ## run search alg
        @time samples, beliefs = mission(seed_val=j, visuals=false, sleep_time=0.0);

        @debug "output correlation matrix:" outputCorMat(beliefs[end])

        ## calculate errors
        metrics[i,j] = calcMetrics(mission, samples, beliefs)
    end
end

## save outputs
# saveOutputs(mission, samples, beliefs, metrics; save_animation=true)
