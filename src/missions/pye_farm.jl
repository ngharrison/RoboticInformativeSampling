# allows using modules defined in any file in project src directory
mod_dir = dirname(Base.active_project()) * "/src/modules"
if mod_dir ∉ LOAD_PATH
    push!(LOAD_PATH, mod_dir)
end

using FileIO: load
using Maps: Map, imgToMap, maps_dir
using Images: gray

using Maps: Map
using SampleCosts: EIGFSampleCost
using Samples: Sample
using SampleCosts: EIGFSampleCost

# this requires a working rospy installation
using ROSInterface: ROSConnection

function pyeFarmMission(; num_samples=4)

    # the topics that will be listened to for measurements
    sub_topics = [
        # Crop height avg and std in frame (excluding wheels)
        ("/rss/gp/crop_height_avg", "/rss/gp/crop_height_std")
    ]

    sampler = ROSConnection(sub_topics)

    # # full bounds
    # lb = [284711.12, 6241319.42]
    # ub = [284802.91, 6241403.93]
    # lb = [284698., 6241343.]
    # ub = [284805., 6241405.]

    # smaller bounds
    # lb = [284729., 6241334.]
    # ub = [284742., 6241348.]
    # lb = [284724., 6241344.]
    # ub = lb .+ 10

    # full space (alt2)
    lb = [284725., 6241345.]
    ub = [284775., 6241395.]

    elev_img = load(maps_dir * "iros_alt2_dem.tif")
    elevMap = imgToMap(gray.(elev_img), lb, ub)

    # # small patch (alt3)
    # lb = [284745., 6241345.]
    # ub = [284760., 6241360.]
    #
    # elev_img = load(maps_dir * "iros_alt3_dem.tif")
    # elevMap = imgToMap(gray.(elev_img), lb, ub)

    prior_maps = [elevMap]

    occupancy = Map(zeros(Bool, 100, 100), lb, ub)

    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations
    n = (7,7) # number of samples in each dimension
    axs_sp = range.(lb, ub, n)
    points_sp = vec(collect.(Iterators.product(axs_sp...)))
    prior_samples = [Sample{Float64}((x, i+length(sampler)), d(x))
                     for (i, d) in enumerate(prior_maps)
                         for x in points_sp if !isnan(d(x))]
    prior_samples = Sample{Float64}[]

    vis(elevMap; points=points_sp)

    num_samples = 30

    sampleCostType = EIGFSampleCost

    ## initialize alg values
    weights = (; μ=1, σ=5e1, τ=1, d=1) # mean, std, dist, prox
    start_locs = [lb .+ 2]

    return Mission(; occupancy,
                   sampler,
                   num_samples,
                   sampleCostType,
                   weights,
                   start_locs,
                   prior_samples)
end
