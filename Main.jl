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

# initialize region
region = Region([0, 0], [1, 1])
peaks = [Peak([0.3, 0.3], 0.03*I, 1.0),
         Peak([0.8, 0.7], 0.008*I, 0.4)]

gt = GaussGT(peaks)
res = [0.01, 0.01]
axis = map((l,r,u)->l:r:u, region.lb, res, region.ub)
# axis = (:).(region.lb, res, region.ub)
points = collect.(collect(Iterators.product(axis...)))
gtMap = gt(points)

dgt = DiscreteGT(gtMap, res)

visualize(dgt, region)

# initialize alg values
weights = [1, 6, 4e-1, 1e-2] # mean, std, dist, prox
x_start = [0.5, 0.2] # starting location

# run search alg
@time samples, belief_model = explore(region, x_start, weights, gt;
                                      show_visuals=true, sleep_time=0.5);

visualize(gt, belief_model, samples, weights, region) # plot stuff
