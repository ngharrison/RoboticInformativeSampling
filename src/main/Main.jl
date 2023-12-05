# set the logging level: Info or Debug
using Logging
global_logger(ConsoleLogger(stderr, Logging.Info))

using AdaptiveSampling: Visualization

using .Visualization: vis

include("../missions/sim.jl")

## initialize data for mission
mission = simMission(num_samples=10)

## run search alg
@time samples, beliefs = mission(
    vis;
    sleep_time=0.0
);


# using Metrics: calcMetrics
# using Outputs: save
#
# ## calculate errors
# metrics = calcMetrics(mission, beliefs, 1)
#
# ## save outputs
# save(mission, samples, beliefs; animation=true)
# save(metrics)
