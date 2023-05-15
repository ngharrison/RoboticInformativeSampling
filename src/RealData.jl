using DelimitedFiles: readdlm
using Statistics: cor

using Environment: imgToMap, MultiMap

# have it run around australia

normalize(a) = a ./ maximum(filter(!isnan, a))

M0 = readdlm("maps/vege_720x360.csv", ',')
M1 = readdlm("maps/topo_720x360.csv", ',')
M2 = readdlm("maps/temp_720x360.csv", ',')
M3 = readdlm("maps/rain_720x360.csv", ',')

M0[M0 .== 99999] .= NaN
M1[M1 .== 99999] .= NaN
M2[M2 .== 99999] .= NaN
M3[M3 .== 99999] .= NaN

australia = (202:258, 587:668)

groundTruth = imgToMap(normalize(M0[australia...]))
multiGroundTruth = MultiMap(groundTruth)

prior_maps = []

push!(prior_maps, imgToMap(normalize(M1[australia...])))
push!(prior_maps, imgToMap(normalize(M2[australia...])))
push!(prior_maps, imgToMap(normalize(M3[australia...])))

occupancy = imgToMap(Matrix{Bool}(isnan.(M0[australia...])))


## initialize alg values
weights = [1, 6, 1, 3e-3] # mean, std, dist, prox
start_loc = [0.8, 0.6] # starting location
num_samples = 50


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

visualize(multiGroundTruth.maps..., prior_maps...;
          titles=["Vegetation", "Elevation", "Ground Temperature", "Rainfall"],
          samples=points_sp)

region = Region(occupancy, multiGroundTruth)
