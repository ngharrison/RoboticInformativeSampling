# using Debugger
using Revise

using Logging
debuglogger = ConsoleLogger(stderr, Logging.Info)
global_logger(debuglogger)

push!(LOAD_PATH, "./") # allows using modules defined in current directory

using Initialization
using Exploration
using Visualization


region = initializeRegion()
gt = createGT()

weights = [1, 6, 4e-1, 1e-2] # mean, std, dist, prox
x_start = [0.5, 0.2] # starting location

@time samples, belief_model = explore(region, x_start, weights, gt; show_visuals=false, sleep_time=0);

visualize(region, samples, gt, belief_model, weights) # plot stuff
