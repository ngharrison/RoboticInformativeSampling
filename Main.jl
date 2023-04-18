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
# using ImageView
# using StaticArrays
# maybe use StructArrays.jl

using Environment
using Exploration
using BeliefModels
using Visualization

## initialize region

# read from image

digi_elev_map = load("maps/arthursleigh_shed_small.tif")

# gui = imshow_gui((500, 500))
# canvas = gui["canvas"]
# imshow(canvas, digi_elev_map)
# Gtk.showall(gui["window"])

res = [0.01, 0.01]
lb = [0, 0]; ub = [1, 1]
axs = (:).(lb, res, ub)
points = collect.(Iterators.product(axs...))

# exclude points within a chosen rectangle
obsMap = zeros(Bool, length.(axs)...)
obsMap[30:75, 35:50] .= true

obs_img = load("maps/obstacles_fieldsouth_220727.tif")
obs_img = imresize(obs_img, Tuple(length.(axs)))
# imshow(obs_img)
obs_map = Matrix{Bool}(reverse(Gray.(obs_img) .== 0, dims=1)')
obsMap = Map(obs_map, (ub.-lb) ./ (size(obs_img).-1))

## initialize ground truth

# simulated
peaks = [Peak([0.3, 0.3], 0.03*I, 1.0),
         Peak([0.8, 0.7], 0.008*I, 0.4)]
ggt = GaussGroundTruth(peaks)
gtMap = Map(ggt(points), res)

## Create prior prior_data

# none -- leave uncommented
data_full = []

if false
    # additive
    push!(data_full, Map(abs.(gtMap .+ 0.1 .* randn(size(gtMap))), res))

    # multiplicative
    push!(data_full, Map(abs.(gtMap .* randn()), res))

    # # both
    # push!(data_full, Map(abs.(gtMap .* randn() + 0.1 .* randn(size(gtMap))), res))

    # # spatial shift
    # t = rand(1:7)
    # push!(data_full, [zeros(size(gtMap,1),t) gtMap[:,1:end-t]]) # shift

    # purely random
    num_peaks = 3
    peaks = [Peak(rand(2).*(ub-lb) .+ lb, 0.02*I, rand())
                for i in 1:num_peaks]
    tggt = GaussGroundTruth(peaks)
    push!(data_full, Map(tggt(points), res))
end

# reduce resolution
# currently all data have the same resolution
h = 20
axs_sp = (:).(1, h, size(gtMap))
prior_data = [Map(d[axs_sp...], res.*h) for d in data_full]

# Calculate correlation coefficients
correlations = [cor(vec(gtMap[axs_sp...]), vec(d)) for d in prior_data]


region = Region(lb, ub, obsMap, gtMap, prior_data)

## initialize alg values
weights = [1, 6, 1, 1e-2] # mean, std, dist, prox
x_start = [0.5, 0.2] # starting location

## run search alg
@time samples, beliefModel = explore(region, x_start, weights;
                                     num_samples=20,
                                     sleep_time=0.5);
                                     # visuals=true,

cov_mat = fullyConnectedCovMat(beliefModel.θ.σ)
correlations = [cov_mat[1,i]/(√cov_mat[1,1]*√cov_mat[i,i]) for i in 2:size(cov_mat, 1)]
