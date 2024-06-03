#!/usr/bin/env julia

using Pkg
Pkg.activate(Base.source_dir() * "/..")

using Logging: global_logger, ConsoleLogger, Info, Debug

using InformativeSampling
using .Maps: Map, generateAxes
using .SampleCosts: EIGFSampleCost
using .Missions: Mission

# this requires a working rospy installation
using .ROSInterface: ROSConnection

function rosMission(; num_samples=4)

    # the topics that will be listened to for measurements
    data_topics = [
        # Crop height avg and std in frame (excluding wheels)
        ("value1", "value2"),
        ("value1", "value2")
    ]

    done_topic = "sortie_finished"
    pub_topic = "latest_sample"

    sampler = ROSConnection(data_topics, done_topic, pub_topic)

    bounds = (lower = [1.0, 1.0], upper = [9.0, 9.0])

    occupancy = Map(zeros(Bool, 100, 100), bounds)

    sampleCostType = EIGFSampleCost

    ## initialize alg values
    weights = (; μ=1, σ=5e3, τ=1, d=1) # mean, std, dist, prox
    start_locs = [[1.0, 1.0]]

    mission = Mission(;
        occupancy,
        sampler,
        num_samples,
        sampleCostType,
        weights,
        start_locs
    )

    return mission
end

#* Run

# set the logging level: Info or Debug
global_logger(ConsoleLogger(stderr, Info))

using InformativeSamplingUtils
using .Visualization: vis
using .DataIO: save

## initialize data for mission
mission = rosMission(num_samples=10)

using PyCall

rospy = pyimport("rospy")
std_msg = pyimport("std_msgs.msg")
sen_msg = pyimport("sensor_msgs.msg")

axs, points = generateAxes(mission.occupancy)
dims = size(mission.occupancy)

quantities = eachindex(mission.sampler)
full_indices = [(p, q) for p in points for q in quantities]

# sets up the publishers
array_publishers, image_publishers = map(("array", "image"),
    (std_msg.Float64MultiArray, sen_msg.Image)) do type_name, type
    map(quantities) do q
        map(("pred", "err")) do name
            rospy.Publisher("/informative_sampling/$(name)_$(type_name)_$(q)",
                type, queue_size=1, latch=true)
        end
    end
end

## run search alg
@time samples, beliefs = mission() do _, _, beliefModel, _, _
    μ, σ = collect.(Iterators.partition.(beliefModel(full_indices), prod(dims)))

    for q in quantities
        for (i, data) in enumerate((μ[q], σ[q]))
            # publish as array
            array = std_msg.Float64MultiArray()
            array.layout.dim = std_msg.MultiArrayDimension.("", collect(dims), 1)
            array.layout.data_offset = 0
            array.data = data
            array_publishers[q][i].publish(array)

            l, h = extrema(data)
            data_scaled = (data .- l) ./ (h - l + eps())

            # map to image
            amount = vec(reverse(permutedims(reshape(data_scaled, dims), (2, 1)), dims=1))

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
        end
    end
end
