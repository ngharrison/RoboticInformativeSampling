using LinearAlgebra: I
using Images: load, imresize, Gray, gray

using Environment: Map, imgToMap, GaussGroundTruth, MultiMap, Peak

# read in elevation
elev_img = load("maps/arthursleigh_shed_small.tif")
elevMap = imgToMap(gray.(elev_img), lb, ub)

obs_img = load("maps/obstacles_fieldsouth_220727.tif")
obs_img_res = imresize(obs_img, size(elev_img))
occ_mat = Matrix{Bool}(Gray.(obs_img_res) .== 0')
occupancy = imgToMap(occ_mat, lb, ub)

## initialize ground truth

# simulated
peaks = [Peak([0.3, 0.3], 0.03*I, 1.0),
         Peak([0.8, 0.7], 0.008*I, 0.4)]
ggt = GaussGroundTruth(peaks)
axs = range.(lb, ub, size(elev_img))
points = collect.(Iterators.product(axs...))
groundTruth = Map(ggt(points), lb, ub)

## Create prior prior_samples

# none -- leave uncommented
prior_maps = []

# additive
push!(prior_maps, Map(abs.(groundTruth .+ 0.1 .* randn(size(groundTruth))), lb, ub))

# multiplicative
push!(prior_maps, Map(abs.(groundTruth .* randn()), lb, ub))

# # both
# push!(prior_maps, Map(abs.(groundTruth .* randn() + 0.1 .* randn(size(groundTruth))), lb, ub))

# # spatial shift
# t = rand(1:7)
# push!(prior_maps, [zeros(size(groundTruth,1),t) groundTruth[:,1:end-t]]) # shift

# purely random
num_peaks = 3
peaks = [Peak(rand(2).*(ub-lb) .+ lb, 0.02*I, rand())
            for i in 1:num_peaks]
tggt = GaussGroundTruth(peaks)
# push!(prior_maps, Map(tggt(points), lb, ub))

multiGroundTruth = MultiMap(groundTruth, Map(tggt(points), lb, ub))

# visualize(multiGroundTruth.maps..., prior_maps...)

## initialize alg values
weights = [1, 6, 1, 1e-2] # mean, std, dist, prox
start_loc = [0.5, 0.2] # starting location

num_samples = 20
