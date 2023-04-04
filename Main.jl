# using Debugger
# using Profile
# using ProfileView
# using Logging
# debuglogger = ConsoleLogger(stderr, Logging.Info)
# global_logger(debuglogger)

using Revise

push!(LOAD_PATH, "./") # allows using modules defined in current directory

using LinearAlgebra
using Environment
using Exploration
using Visualization

# maybe use StructArrays.jl

## initialize region

res = [0.01, 0.01]
lb = [0, 0]; ub = [1, 1]
axs = (:).(lb, res, ub)
points = collect.(Iterators.product(axs...))

# exclude points within a chosen rectangle
obstacle_lb = (30, 35).*res
obstacle_ub = (75, 50).*res
obsMap = reduce(.&, [obstacle_lb[i] .<= getindex.(points, i) .<= obstacle_ub[i]
                     for i in 1:length(obstacle_lb)])
obsMap = Map(obsMap, res)

## initialize ground truth
peaks = [Peak([0.3, 0.3], 0.03*I, 1.0),
         Peak([0.8, 0.7], 0.008*I, 0.4)]
ggt = GaussGT(peaks)
gtMap = Map(ggt(points), res)

region = Region(lb, ub, obsMap, gtMap)

visualize(gtMap, region)

## initialize alg values
weights = [1, 6, 4e-1, 1e-2] # mean, std, dist, prox
x_start = [0.5, 0.2] # starting location

## run search alg
@time samples, beliefModel = explore(region, x_start, weights;
                                     num_samples=10,
                                     show_visuals=false,
                                     sleep_time=0.0);
