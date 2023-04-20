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
using Images
# using StaticArrays
# maybe use StructArrays.jl

using Environment
using Exploration
using Samples
using BeliefModels
using Visualization

## initialize region

lb = [0, 0]; ub = [1, 1]

# read from image

# read in elevation
elev_img = load("maps/arthursleigh_shed_small.tif")
elevMap = imgToMap(gray.(elev_img), lb, ub)

# gui = imshow_gui((500, 500))
# canvas = gui["canvas"]
# imshow(canvas, digi_elev_map)
# Gtk.showall(gui["window"])

obs_img = load("maps/obstacles_fieldsouth_220727.tif")
obs_img = imresize(obs_img, size(elev_img))
# imshow(obs_img)
# switch from row-column to x-y format
occ_mat = Matrix{Bool}(Gray.(obs_img) .== 0')
occMap = imgToMap(occ_mat, lb, ub)

## initialize ground truth

# simulated
peaks = [Peak([0.3, 0.3], 0.03*I, 1.0),
         Peak([0.8, 0.7], 0.008*I, 0.4)]
ggt = GaussGroundTruth(peaks)
axs = range.(lb, ub, size(elev_img))
points = collect.(Iterators.product(axs...))
gtMap = Map(ggt(points), lb, ub)

## Create prior prior_samples

# none -- leave uncommented
prior_maps = []

# additive
push!(prior_maps, Map(abs.(gtMap .+ 0.1 .* randn(size(gtMap))), lb, ub))

# multiplicative
push!(prior_maps, Map(abs.(gtMap .* randn()), lb, ub))

# # both
# push!(prior_maps, Map(abs.(gtMap .* randn() + 0.1 .* randn(size(gtMap))), lb, ub))

# # spatial shift
# t = rand(1:7)
# push!(prior_maps, [zeros(size(gtMap,1),t) gtMap[:,1:end-t]]) # shift

visualize(gtMap, prior_maps...)

# purely random
num_peaks = 3
peaks = [Peak(rand(2).*(ub-lb) .+ lb, 0.02*I, rand())
            for i in 1:num_peaks]
tggt = GaussGroundTruth(peaks)
push!(prior_maps, Map(tggt(points), lb, ub))

# sample sparsely from the prior maps
# currently all data have the same sample numbers and locations
n = (5,5) # number of samples in each dimension
axs_sp = range.(lb, ub, n)
points_sp = vec(collect.(Iterators.product(axs_sp...)))
prior_samples = [Sample((x, i+1), d(x)) for (i, d) in enumerate(prior_maps) for x in points_sp]

# Calculate correlation coefficients
correlations = [cor(gtMap.(points_sp), d.(points_sp)) for d in prior_maps]


region = Region(occMap, gtMap)

## initialize alg values
weights = [1, 6, 1, 1e-2] # mean, std, dist, prox
x_start = [0.5, 0.2] # starting location

## run search alg
@time samples, beliefModel = explore(region, x_start, weights;
                                     num_samples=20,
                                     prior_samples,
                                     # visuals=true,
                                     sleep_time=0.0);

cov_mat = fullyConnectedCovMat(beliefModel.θ.σ)
correlations = [cov_mat[i,1]/√(cov_mat[1,1]*cov_mat[i,i]) for i in 2:size(cov_mat, 1)]
