#!/usr/bin/env julia
# should be set and run as an executable

using Pkg
Pkg.activate(Base.source_dir() * "/..")

using RobotOS

using PyCall
rospy = pyimport("rospy")
geometry_msg = pyimport("geometry_msgs.msg")
sensor_msg = pyimport("sensor_msgs.msg")

using Images: colorview, RGBA
using Plots

# 6167694.42548 229994.046421
const my_xlim = [229989, 230189]
const my_ylim = [6167689, 6167889]

function plotStuff()
    pose_array = rospy.wait_for_message("munch/animal_poses", geometry_msg.PoseArray)

    x = [pose.position.x for pose in pose_array.poses]
    y = [pose.position.y for pose in pose_array.poses]

    image = rospy.wait_for_message("munch/nutrient_map", sensor_msg.Image)

    im = permutedims(reshape(Vector{UInt8}(image.data), (4, image.width, image.height)), (1,3,2))
    ima = colorview(RGBA, im/255)

    p = plot(my_xlim, my_ylim, ima)
    scatter!(
        x, y;
        xlim=my_xlim,
        ylim=my_ylim,
        legend=false,
        framestyle=:box,
        aspect_ratio=:equal,
        color=:lightblue
    )
    p |> gui
end

function main()
    init_node("pose_plotter")
    loop_rate = Rate(20.0)
    while !is_shutdown()
        plotStuff()
        rossleep(loop_rate)
    end

end

main()
