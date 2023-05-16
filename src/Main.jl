# This file is used as the general scripting area
# and launching point for running the parts of the
# system. The main parts are initializing data,
# running the algorithm, and visualizing results.

# allows using modules defined in any file in project src directory
push!(LOAD_PATH, dirname(Base.active_project()) * "/src")

using Initialization: simData, realData
using BeliefModels: correlations
using Visualization: visualize
using Exploration: explore

## initialize region

# initialize data use simData or realData for this
region, start_loc, weights, num_samples, prior_samples = realData()

## run search alg
@time samples, beliefModel = explore(region, start_loc, weights;
                                     num_samples,
                                     prior_samples,
                                     visuals=true,
                                     sleep_time=0.0);

println()
println("Correlations: $(round.(correlations(beliefModel), digits=3))")
