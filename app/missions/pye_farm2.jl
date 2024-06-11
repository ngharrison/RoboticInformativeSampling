
using Logging: global_logger, ConsoleLogger, Info, Debug
using FileIO: load
using Images: gray

using InformativeSampling
using .Maps: Map, generateAxes, getBounds
using .SampleCosts: EIGF, DistScaledEIGF
using .Samples: Sample, MapsSampler
using .Missions: Mission
using .BeliefModels: BeliefModel

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
        "/rss/silios/ndvi_bw_gradient_image" # TODO check this
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
        prior_samples = [Sample{Float64}((x, i + 1), d(x))
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
    sampleCostType=DistScaledEIGF,
    use_priors=false,
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


#* Publish maps

using PyCall

rospy = pyimport("rospy")
std_msg = pyimport("std_msgs.msg")
sen_msg = pyimport("sensor_msgs.msg")

axs, points = generateAxes(mission.occupancy)
dims = size(mission.occupancy)

quantities = eachindex(mission.sampler)

# sets up the publishers
array_publishers, image_publishers = map(("array", "image"),
    (std_msg.Float64MultiArray, sen_msg.Image)) do type_name, type
    map(quantities) do q
        map(("pred", "err")) do name
            rospy.Publisher("/informative_sampling/$(name)_$(type_name)_$(q)",
                type, queue_size=1, latch=false)
        end
    end
end

M = mission
other_bm = BeliefModel(filter(s->s.x[2]==2, samples), getBounds(M.occupancy);
                       M.noise, M.kernel)
bm_vals = map(enumerate((beliefs[end], other_bm))) do (i, bm)
    bm(tuple.(vec(points), i))
end

for q in quantities
    for (i, data) in enumerate(bm_vals[q])
        # publish as array
        array = std_msg.Float64MultiArray()
        array.layout.dim = std_msg.MultiArrayDimension.("", collect(dims), 1)
        array.layout.data_offset = 0
        array.data = data
        array_publishers[q][i].publish(array)
        rospy.sleep(.01) # allow the publisher time to publish

        l, h = extrema(data)
        data_scaled = (data .- l) ./ (h - l + eps())

        # map to image
        amount = vec(reverse(permutedims(reshape(data_scaled, dims), (2, 1)), dims=1))

        # Main.@infiltrate

        # convert to byte values and repeat each 4 times for rgba format
        byte_data = round.(UInt8, 255 .* amount)
        img_data = pybytes([x for x in byte_data for _ in 1:4])

        env_height, env_width = dims

        # publish as image
        image = sen_msg.Image()
        image.header.stamp = rospy.Time.now()
        image.header.frame_id = "env_image"
        image.height = env_height
        image.width = env_width
        image.encoding = "rgba8"
        image.is_bigendian = 1
        image.step = env_width * 4
        image.data = img_data
        image_publishers[q][i].publish(image)
        rospy.sleep(.01) # allow the publisher time to publish
    end
end
