
using Logging: global_logger, ConsoleLogger, Info, Debug
using FileIO: load
using Images: gray

using AdaptiveSampling

using .Maps: Map, imgToMap, maps_dir
using .SampleCosts: EIGFSampleCost
using .Samples: Sample
using .SampleCosts: EIGFSampleCost
using .Missions: Mission
using .Visualization: vis

# this requires a working rospy installation
using .ROSInterface: ROSConnection

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

#* Run

# set the logging level: Info or Debug
global_logger(ConsoleLogger(stderr, Debug))

using .Visualization: vis
using .Outputs: save

## initialize data for mission
mission = pyeFarmMission()

## run search alg
@time samples, beliefs = mission(
    sleep_time=0.0
) do M, samples, beliefModel, sampleCost, new_loc
    vis(beliefModel, samples, new_loc, M.occupancy, 1)
    save(M, samples, beliefModel)
end;
save(mission, samples, beliefs)

## save outputs
# saveBeliefMapToPng(beliefs[end], mission.occupancy)
