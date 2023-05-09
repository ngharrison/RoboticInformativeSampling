# This file is used as the general scripting area
# and launching point for running the parts of the
# system. The main parts are initializing data,
# running the algorithm, and visualizing results.

# using Debugger
# using Profile
# using ProfileView
# using StatProfilerHTML

using Revise
push!(LOAD_PATH, "./") # allows using modules defined in current directory
Base.active_repl.options.iocontext[:displaysize] = (20, 70) # limit lines printed out
# Base.active_repl.options.iocontext[:displaysize] = displaysize(stdout) # set back to default

using LinearAlgebra
using Statistics
# using StaticArrays

using Environment
using Samples
using BeliefModels
using Visualization
using Exploration

## initialize region

lb = [0, 0]; ub = [1, 1]

# initialize data
# TODO make these into functions
data_type = :sim
if data_type === :sim
    include("SimData.jl")
elseif data_type === :real
    include("RealData.jl")
end

ros_data = initRos()

# sample sparsely from the prior maps
# currently all data have the same sample numbers and locations
n = (5,5) # number of samples in each dimension
axs_sp = range.(lb, ub, n)
points_sp = vec(collect.(Iterators.product(axs_sp...)))
prior_samples = [Sample((x, i+length(multiGroundTruth)), d(x))
                 for (i, d) in enumerate(prior_maps)
                     for x in points_sp if !isnan(d(x))]

# Calculate correlation coefficients
[cor(groundTruth.(points_sp), d.(points_sp)) for d in prior_maps]


region = Region(occupancy, multiGroundTruth)

## run search alg
@time samples, beliefModel = explore(region, start_loc, weights;
                                     num_samples,
                                     prior_samples,
                                     ros_data,
                                     visuals=true,
                                     sleep_time=0.0);

@show correlations(beliefModel);
