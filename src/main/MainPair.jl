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

@time for priors in [(0,0,0), (1,1,1)]
    ## initialize data for mission
    priors=(0,0,0)
    mission = ausMission(priors=collect(Bool, priors))
    # empty!(mission.prior_samples)

    # s = [simMission(priors=Bool[1,1,1]; seed_val)[2].^2 for seed_val in 1:10]
    # mean(s)
    # std([.√a for a in s])

    ## run search alg
    @time samples, beliefs = mission(visuals=true, sleep_time=0.0);
    @debug "output correlation matrix:" outputCorMat(beliefs[end])
    # save(mission, samples, beliefs; animation=true)

    ## calculate errors
    metrics = calcMetrics(mission, beliefs, 1)

    ## save outputs
    save(metrics; file_name="aus_ave/metrics_$(join(priors))")
    save(mission, samples, beliefs; file_name="aus_ave/mission_$(join(priors))")
end
