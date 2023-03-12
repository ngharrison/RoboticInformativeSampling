# using Debugger
# using Profile
# using ProfileView
# using Logging
# debuglogger = ConsoleLogger(stderr, Logging.Info)
# global_logger(debuglogger)

using Revise

push!(LOAD_PATH, "./") # allows using modules defined in current directory

using LinearAlgebra
using Initialization
using Exploration
using Visualization


region = initializeRegion()
peaks = [Peak([0.3, 0.3], 0.03*I, 1.0),
         Peak([0.8, 0.7], 0.008*I, 0.4)]
gt = GaussGT(peaks)
visualize(region, gt)

weights = [1, 6, 4e-1, 1e-2] # mean, std, dist, prox
x_start = [0.5, 0.2] # starting location

@time samples, belief_model = explore(region, x_start, weights, gt;
                                      show_visuals=false, sleep_time=0.0);

visualize(region, samples, gt, belief_model, weights) # plot stuff
