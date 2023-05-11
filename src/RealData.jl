using DelimitedFiles

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

# visualize(multiGroundTruth.maps..., prior_maps...)
