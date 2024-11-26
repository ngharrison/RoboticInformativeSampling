
using Logging: global_logger, ConsoleLogger, Info, Debug
using FileIO: load
using Images: gray

using GridMaps: GridMap

using InformativeSampling
using .SampleCosts: EIGF
using .Samples: Sample
using .Missions: Mission

# this requires a working rospy installation
using .ROSInterface: ROSSampler

using InformativeSamplingUtils
using .Visualization: vis
using .DataIO: maps_dir, imgToMap

function pyeFarmMission(; num_samples=4)

    # the topics that will be listened to for measurements
    data_topics = [
        # Crop height avg in frame (excluding wheels)
        "/rss/gp/crop_height_avg"
    ]

    done_topic = "sortie_finished"
    pub_topic = "latest_sample"

    sampler = ROSSampler(data_topics, done_topic, pub_topic)

    # # full bounds
    # lower=[284711.12, 6241319.42]
    # upper=[284802.91, 6241403.93]
    # bounds = (; lower, upper)
    # lower=[284698.0, 6241343.0]
    # upper=[284805.0, 6241405.0]
    # bounds = (; lower, upper)

    # # smaller bounds
    # lower=[284729.0, 6241334.0]
    # upper=[284742.0, 6241348.0]
    # bounds = (; lower, upper)
    # lower=[284724.0, 6241344.0]
    # upper=bounds.lower .+ 10
    # bounds = (; lower, upper)

    # full space (alt2)
    lower=[284725.0, 6241345.0]
    upper=[284775.0, 6241395.0]
    bounds = (; lower, upper)

    elev_img = load(maps_dir * "iros_alt2_dem.tif")
    elevMap = imgToMap(gray.(elev_img), bounds)

    # # small patch (alt3)
    # lower=[284745.0, 6241345.0]
    # upper=[284760.0, 6241360.0]
    # bounds = (; lower, upper)
    #
    # elev_img = load(maps_dir * "iros_alt3_dem.tif")
    # elevMap = imgToMap(gray.(elev_img), bounds)

    prior_maps = [elevMap]

    occupancy = GridMap(zeros(Bool, 100, 100), bounds)

    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations
    n = (7,7) # number of samples in each dimension
    axs_sp = range.(bounds..., n)
    points_sp = vec(collect.(Iterators.product(axs_sp...)))
    prior_samples = [Sample{Float64}((x, i+length(sampler)), d(x))
                     for (i, d) in enumerate(prior_maps)
                         for x in points_sp if !isnan(d(x))]
    prior_samples = Sample{Float64}[]

    sampleCostType = EIGF

    ## initialize alg values
    weights = (; μ=1, σ=5e1, τ=1, d=1) # mean, std, dist, prox
    start_locs = [bounds.lower .+ 2]

    mission = Mission(;
        occupancy,
        sampler,
        num_samples,
        sampleCostType,
        weights,
        start_locs,
        prior_samples
    )

    return mission, prior_maps
end

#* Run

# set the logging level: Info or Debug
global_logger(ConsoleLogger(stderr, Debug))

using .DataIO: save

## initialize data for mission
mission, prior_maps = pyeFarmMission()

vis(prior_maps[1]; points=first.(getfield.(mission.prior_samples, :x)))

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
