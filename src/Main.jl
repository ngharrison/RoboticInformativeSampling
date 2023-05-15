# This file is used as the general scripting area
# and launching point for running the parts of the
# system. The main parts are initializing data,
# running the algorithm, and visualizing results.

push!(LOAD_PATH, "./") # allows using modules defined in current directory

using BeliefModels: correlations
using Visualization: visualize
using Exploration: explore

## initialize region

lb = [0, 0]; ub = [1, 1]

# initialize data
# TODO make these into functions
data_type = :real
if data_type === :sim
    include("SimData.jl")
elseif data_type === :real
    include("RealData.jl")
end

## run search alg
@time samples, beliefModel = explore(region, start_loc, weights;
                                     num_samples,
                                     prior_samples,
                                     visuals=true,
                                     sleep_time=0.0);

println()
println("Correlations: $(round.(correlations(beliefModel), digits=3))")
