#!/usr/bin/env julia

module ROSInterface

using RobotOS
@rosimport std_msgs.msg: Float32Msg
@rosimport geometry_msgs.msg: Pose
rostypegen(@__MODULE__)
using .std_msgs.msg
using .geometry_msgs.msg

using Environment

export main

const sub_nodes = [
    "/GP_Data/avg_NDVI",
    "/GP_Data/avg_crop_height",
    "/GP_Data/cover_NDVI"
]

# needs
# - data values combined so we can be sure we have all new values at once
#   could be put into 3Vector on other side
# - knowledge of new location or that the commanded location is reached
# - similar interface between gazebo sim and swagbot code
# - decide between message and service

function callback(msg::Float32Msg)
    # new value received

    # check robot location
    # TODO need to figure out the localization for swagbot
    loc = rospy.wait_for_message("/robot_location???", Pose, timeout=5)

    # create sample
    # TODO get node_name
    quantity = findfirst(==(node_name), sub_nodes)

    ros_data.values[quantity] = msg
end

function publishNextLocation(pub_obj::Publisher{Pose}, new_loc::Location)
    next_sample = Pose(next_loc...)

    # for swagbot
    msg_goalpose.position.x = new_loc[1];
    msg_goalpose.position.y = new_loc[2];
    msg_goalpose.position.z = 0;
    # TODO need to actually set these as well, from the path
    msg_goalpose.orientation.w = 0;
    msg_goalpose.orientation.x = 0;
    msg_goalpose.orientation.y = 0;
    msg_goalpose.orientation.z = 0;

    publish(pub_obj, next_sample)
end

function initRos()
    init_node("adaptive_sampling")

    # this passes just the location, no quantity id
    pub = Publisher{Pose}("latest_sample", queue_size=10)

    subs = [Subscriber{Float32Msg}(sub, callback, (ros_data,), queue_size=10)
            for sub in sub_nodes]

    return (; pub, subs, values)
end

if !isinteractive()
    main()
end

end
