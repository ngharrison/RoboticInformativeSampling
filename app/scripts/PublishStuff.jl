
using InformativeSampling
using .Maps
using .BeliefModels

using InformativeSamplingUtils
using .Visualization
using .DataIO

using FileIO

#* Publish maps

using PyCall

rospy = pyimport("rospy")
std_msg = pyimport("std_msgs.msg")
sen_msg = pyimport("sensor_msgs.msg")

quantities = 1:2

rospy.init_node("publishing_maps")

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

rospy.sleep(.1)


name = "50x50"
file_name = output_dir * "pye_farm_trial2/named/" * name * output_ext
data = load(file_name)
noise = (value=0.1, learned=false)

lower = [284725.0, 6241345.0]
upper = [284775.0, 6241395.0]
bounds = (; lower, upper)
occupancy = Map(zeros(100, 100), bounds)

axs, points = generateAxes(occupancy)
dims = size(occupancy)

rospy.sleep(.1)

for q in quantities
    bm = BeliefModel(filter(s->s.x[2]==q, data["samples"]), bounds; noise)

    pred, err = bm(tuple.(vec(points), q))

    for (i, data) in enumerate((pred, err))

        # map to image
        data_flipped = vec(reverse(permutedims(reshape(data, dims), (2, 1)), dims=1))

        # publish as array
        array = std_msg.Float64MultiArray()
        array.layout.dim = std_msg.MultiArrayDimension.("", collect(dims), 1)
        array.layout.data_offset = 0
        array.data = data_flipped

        l, h = extrema(data_flipped)
        data_scaled = (data_flipped .- l) ./ (h - l + eps())

        # convert to byte values and repeat each 4 times for rgba format
        byte_data = round.(UInt8, 255 .* data_scaled)
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

        array_publishers[q][i].publish(array)
        image_publishers[q][i].publish(image)
        rospy.sleep(.05) # allow the publishers time to publish
    end
end

rospy.sleep(.1)
