# This file is used as the general scripting area
# and launching point for running the parts of the
# system. The main parts are initializing data,
# running the algorithm, and visualizing results.

# using Debugger
# using Profile
# using ProfileView

using Revise
push!(LOAD_PATH, "./") # allows using modules defined in current directory
Base.active_repl.options.iocontext[:displaysize] = (20, 70) # limit lines printed out
# Base.active_repl.options.iocontext[:displaysize] = displaysize(stdout) # set back to default

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
obsMap = zeros(Bool, length.(axs)...)
obsMap[30:75, 35:50] .= true
obsMap = Map(obsMap, res)

## initialize ground truth
peaks = [Peak([0.3, 0.3], 0.03*I, 1.0),
         Peak([0.8, 0.7], 0.008*I, 0.4)]
ggt = GaussGroundTruth(peaks)
gtMap = Map(ggt(points), res)

region = Region(lb, ub, obsMap, gtMap)

## initialize alg values
weights = [1, 6, 1, 1e-2] # mean, std, dist, prox
x_start = [0.5, 0.2] # starting location

## run search alg
@time samples, beliefModel = explore(region, x_start, weights;
                                     num_samples=20,
                                     sleep_time=0.5);
                                     # visuals=true,
