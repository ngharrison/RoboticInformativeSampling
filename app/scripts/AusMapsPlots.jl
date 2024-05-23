
using Random: seed!
using LinearAlgebra: norm
using Statistics: cor
using DelimitedFiles: readdlm
using Plots

using InformativeSampling

using .Maps: Map, getBounds
using .Samples: Sample, MapsSampler, selectSampleLocation

using InformativeSamplingUtils
using .Visualization: visualize
using .DataIO: maps_dir, output_dir, imgToMap, spatialAve, normalize

#* data
seed_val=0; priors=Bool[1,1,1];

seed!(seed_val)

file_names = [
    "vege_ave_aus.csv",
    "topo_ave_aus.csv",
    "temp_ave_aus.csv",
    "rain_ave_aus.csv"
]

images = readdlm.(maps_dir .* file_names, ',')

ims_sm = spatialAve.(images, 0)

bounds = (lower = [0.0, 0.0], upper = [1.0, 1.0])

map0 = imgToMap(normalize(ims_sm[1]), bounds)

# image = readdlm(maps_dir * "vege_ave_nsw.csv", ',')
# mapx = imgToMap(image, bounds)
# heatmap(mapx')
# savefig("temp.png")

# scatter!([3211, 3211, 3310, 3310], 1800 .- [1141, 1240, 1141, 1240])
sampler = MapsSampler(map0)

prior_maps = [imgToMap(normalize(img), bounds) for img in ims_sm[2:end]]

occupancy = imgToMap(Matrix{Bool}(reduce(.|, [isnan.(i)
                                              for i in ims_sm])), bounds)

# sample sparsely from the prior maps
# currently all data have the same sample numbers and locations
n = (5,5) # number of samples in each dimension
axs_sp = range.(bounds..., n)
points_sp = vec(collect.(Iterators.product(axs_sp...)))
prior_samples = [Sample((x, i+length(sampler)), d(x))
                 for (i, d) in enumerate(prior_maps[priors])
                     for x in points_sp if !isnan(d(x))]

# maximize minimum distance between samples
points_sp = Vector{Float64}[]
sampleCost = x -> occupancy(x) ? Inf : -minimum(norm(loc - x) for loc in points_sp; init=Inf)
for _ in 1:25
    x = selectSampleLocation(sampleCost, getBounds(occupancy)...)
    push!(points_sp, x)
end
prior_samples = [Sample((x, i+length(sampler)), d(x))
                 for (i, d) in enumerate(prior_maps[priors])
                     for x in points_sp if !isnan(d(x))]

# Calculate correlation coefficients
[cor(map0.(points_sp), d.(points_sp)) for d in prior_maps]
[cor(vec(map0[.!occupancy]), vec(d[.!occupancy])) for d in prior_maps]
# scatter(vec(map0[.!occupancy]), [vec(d[.!occupancy]) for d in prior_maps], layout=3)

# Full map correlations:
#
# 3-element Vector{Float64}:
#   0.03648870556448797
#  -0.3304252405503303
#   0.07525846578605022


#* plot maps
axs = range.(bounds..., size(occupancy))
p1 = heatmap(axs..., map0', title="Vegetation (QOI)", ticks=false, framestyle=:none)
p2 = heatmap(axs..., prior_maps[1]', title="Elevation", ticks=false)
scatter!(first.(points_sp), last.(points_sp);
         framestyle=:none,
         label="Samples",
         color=:green,
         legend=(0.15, 0.87),
         markersize=6)
p3 = heatmap(axs..., prior_maps[2]', title="Ground Temperature", ticks=false)
scatter!(first.(points_sp), last.(points_sp);
         framestyle=:none,
         label="Samples",
         color=:green,
         legend=(0.15, 0.87),
         markersize=6)
p4 = heatmap(axs..., prior_maps[3]', title="Rainfall", ticks=false)
scatter!(first.(points_sp), last.(points_sp);
         framestyle=:none,
         label="Samples",
         color=:green,
         legend=(0.15, 0.87),
         markersize=6)
cy = 0:.001:1
cbar = heatmap([0], cy, reshape(cy, :, 1),
               # aspect_ratio=1/100,
               xticks=false,
               yticks=0:0.5:1,
               mirror=true
               )
plot(p1, p2, p3, p4, cbar,
     layout=@layout([grid(2,2){0.95w} c]),
     # clim=(0,1),
     colorbar=false,
     size=(1000, 800),
     titlefontsize=24,
     tickfontsize=20,
     legendfontsize=14,
     margin=4Plots.mm
)

savefig(output_dir * "paper/aus_ave_data_maps.png")
