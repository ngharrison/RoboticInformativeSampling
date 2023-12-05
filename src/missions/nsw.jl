using DelimitedFiles: readdlm
using Statistics: cor
using Random: seed!

using AdaptiveSampling: Maps, Samples, SampleCosts, Missions, Visualization

using .Maps: Map, imgToMap, maps_dir
using .Samples: Sample, MapsSampler
using .SampleCosts: EIGFSampleCost
using .Missions: Mission
using .Visualization: vis

include("../utils/utils.jl")

function nswMission(; seed_val=0, num_samples=30, priors=Bool[1,1,1])
    # have it run around australia

    seed!(seed_val)

    file_names = [
        "vege_ave_nsw.csv",
        "topo_ave_nsw.csv",
        "temp_ave_nsw.csv",
        "rain_ave_nsw.csv"
    ]

    images = readdlm.(maps_dir .* file_names, ',')

    ims_sm = images

    lb = [0.0, 0.0]; ub = [1.0, 1.0]

    map0 = imgToMap(normalize(ims_sm[1]), lb, ub)
    sampler = MapsSampler(map0)

    prior_maps = [imgToMap(normalize(img), lb, ub) for img in ims_sm[2:end]]

    occupancy = imgToMap(Matrix{Bool}(reduce(.|, [isnan.(i)
                                                  for i in ims_sm])), lb, ub)

    sampleCostType = EIGFSampleCost

    ## initialize alg values
    # weights = [1e-1, 6, 5e-1, 3e-3] # mean, std, dist, prox
    # weights = (; μ=1, σ=5e3, τ=1, d=1) # others
    weights = (; μ=1, σ=5e2, τ=1, d=1) # others
    start_locs = [[1.0, 0.0]] # starting locations


    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations

    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations
    n = (5,5) # number of samples in each dimension
    axs_sp = range.(lb, ub, n)
    points_sp = vec(collect.(Iterators.product(axs_sp...)))
    prior_samples = [Sample((x, i+length(sampler)), d(x))
                     for (i, d) in enumerate(prior_maps[priors])
                         for x in points_sp if !isnan(d(x))]

    # Calculate correlation coefficients
    [cor(map0.(points_sp), d.(points_sp)) for d in prior_maps]
    [cor(vec(map0[.!occupancy]), vec(d[.!occupancy])) for d in prior_maps]
    # scatter(vec(map0[.!occupancy]), [vec(d[.!occupancy]) for d in prior_maps], layout=3)

    vis(sampler.maps..., prior_maps...;
              titles=["Vegetation", "Elevation", "Ground Temperature", "Rainfall"],
              points=points_sp)

    return Mission(; occupancy,
                   sampler,
                   num_samples,
                   sampleCostType,
                   weights,
                   start_locs,
                   prior_samples)


end

