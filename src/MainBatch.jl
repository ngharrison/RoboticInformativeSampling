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

mission_peaks = [3,3,4,4,5,5]
num_runs = 3
metrics = Array{Any, 2}(undef, (length(mission_peaks), num_runs))
# pick all the prior data combinations
@time for priors in Iterators.product(fill(0:1,3)...)
    for (i, num_peaks) in enumerate(mission_peaks)
        ## initialize data for mission
        mission = simMission(; seed_val=i, num_peaks, priors=collect(Bool, priors))
        for j in 1:num_runs
            ## run search alg
            @time samples, beliefs = mission(seed_val=j, visuals=false, sleep_time=0.0);
            @debug "output determination matrix:" outputCorMat(beliefs[end]).^2
            # save(mission, samples, beliefs; animation=true)
            ## calculate errors
            metrics[i,j] = calcMetrics(mission, beliefs, 1)
        end
    end
    ## save outputs
    save(metrics; file_name="batch/metrics_$(join(priors))")
end
