
using Logging: global_logger, ConsoleLogger, Info, Debug
using FileIO: load
using Images: gray

using InformativeSampling
using .Maps: Map, generateAxes, getBounds
using .SampleCosts: EIGF, DistScaledEIGF
using .Samples: Sample, MapsSampler
using .Missions: Mission

# this requires a working rospy installation
using .ROSInterface: ROSConnection

using InformativeSamplingUtils
using .Visualization: vis
using .DataIO: maps_dir, imgToMap

function pyeFarmMission(; num_samples=4,
    sampleCostType=DistScaledEIGF,
    use_priors=false,
    start_locs=[])

    # the topics that will be listened to for measurements
    data_topics = [
        # Crop height avg in frame (excluding wheels)
        "/rss/gp/crop_height_avg"
    ]

    done_topic = "sortie_finished"
    pub_topic = "latest_sample"

    sampler = ROSConnection(data_topics, done_topic, pub_topic)

    # 50x50 meter space (alt2)
    lower=[284725.0, 6241345.0]
    upper=[284775.0, 6241395.0]
    bounds = (; lower, upper)

    elev_img = load(maps_dir * "dem_50x50.tif")
    elevMap = imgToMap(gray.(elev_img), bounds)

    occupancy = Map(zeros(Bool, 100, 100), bounds)

    # # to test
    # p = let
    #     name = "100samples_50x50_grid"
    #     file_name = output_dir * "pye_farm_trial_named/" * name * output_ext
    #     data = load(file_name)
    #     samples = data["samples"]
    #
    #     bm = BeliefModel(samples, bounds)
    #     p, s = produceMaps(bm, occupancy)
    #     # vis(p)
    #     p
    # end
    #
    # sampler = MapsSampler(p)

    # # 15x15 meter sub-patch (alt3)
    # lower=[284745.0, 6241345.0]
    # upper=[284760.0, 6241360.0]
    # bounds = (; lower, upper)
    #
    # elev_img = load(maps_dir * "dem_15x15.tif")
    # elevMap = imgToMap(gray.(elev_img), bounds)

    prior_maps = [elevMap]

    prior_samples = Sample{Float64}[]

    if use_priors
        # sample sparsely from the prior maps
        # currently all data have the same sample numbers and locations
        n = (7, 7) # number of samples in each dimension
        axs_sp = range.(bounds..., n)
        points_sp = vec(collect.(Iterators.product(axs_sp...)))
        prior_samples = [Sample{Float64}((x, i + length(sampler)), d(x))
                         for (i, d) in enumerate(prior_maps)
                         for x in points_sp if !isnan(d(x))]
    end

    ## initialize alg values
    weights = (; μ=1, σ=5e1, τ=1, d=1) # mean, std, dist, prox

    noise = (value=0.1, learned=true)

    mission = Mission(;
        occupancy,
        sampler,
        num_samples,
        sampleCostType,
        weights,
        start_locs,
        prior_samples,
        noise
    )

    return mission, prior_maps
end

#* Run

# set the logging level: Info or Debug
global_logger(ConsoleLogger(stderr, Debug))

using .DataIO: save

## initialize data for mission
mission, prior_maps = pyeFarmMission(
    num_samples=30,
    sampleCostType=EIGF,
    use_priors=true,
    start_locs=[]
)

vis(prior_maps[1]; points=first.(getfield.(mission.prior_samples, :x)))

## run search alg
@time samples, beliefs = mission(
    sleep_time=0.0
) do M, samples, beliefModel, sampleCost, new_loc
    vis(M, samples, beliefModel, sampleCost, new_loc)
    save(M, samples, beliefModel)
end;
save(mission, samples, beliefs)
