#!/usr/bin/env julia

module ROSInterface

using RobotOS
@rosimport std_msgs.msg: Bool, Float64
@rosimport geometry_msgs.msg: Pose, Point, Quaternion
rostypegen(@__MODULE__)
using .std_msgs.msg
using .geometry_msgs.msg

using PyCall

using Rotations: QuatRotation, RotZ, params

using Environment: Location, Index

"""
Stores information for communicating with Swagbot.
"""
mutable struct ROSConnection
    sub_nodes
    pub_next_sample
    pub_finished
    sortie_finished
end

"""
Constructor.
"""
function ROSConnection(sub_nodes)
    # initialize this node with its name
    init_node("adaptive_sampling")

    # this will pass the full goal pose, no quantity id
    pub_next_sample = Publisher{Pose}("latest_sample", queue_size=10)
    pub_finished = Publisher{BoolMsg}("sortie_finished", queue_size=10)

    # create connection object
    rosConnection = ROSConnection(sub_nodes, pub_next_sample, pub_finished, false)

    # subscriber to check if sortie is finished
    sortie_sub = Subscriber{BoolMsg}("sortie_finished", saveFinished,
                                        (rosConnection,), queue_size=10)

    return rosConnection
end

# give it a length and indices
Base.keys(R::ROSConnection) = keys(R.sub_nodes)
Base.length(R::ROSConnection) = length(R.sub_nodes)

"""
Returns a vector of values from the sample location.
"""
function (R::ROSConnection)(new_loc::Location)
    publishNextLocation(R.pub_next_sample, new_loc)

    # wait, check each second
    while !(R.sortie_finished || is_shutdown())
        rossleep(1)
    end

    # reset for next time
    publish(R.pub_finished, BoolMsg(false))

    # get values
    rospy = pyimport("rospy")
    std_msg = pyimport("std_msgs.msg")
    values = [rospy.wait_for_message(node, std_msg.Float64, timeout=5).data
              for node in R.sub_nodes]

    return values
end

"""
Returns a single value from the sample location of the chosen quantity.

Currently will be unused.
"""
function (R::ROSConnection)(new_index::Index)
    loc, quantity = new_index
    publishNextLocation(R.pub_next_sample, loc)

    # wait, check each second
    while !(R.sortie_finished || is_shutdown())
        rossleep(1)
    end

    # reset for next time
    publish(R.pub_finished, BoolMsg(false))

    # get values
    rospy = pyimport("rospy")
    std_msg = pyimport("std_msgs.msg")
    values = rospy.wait_for_message(R.sub_nodes[quantity], std_msg.Float64, timeout=5)

    return values
end

"""
Callback function to save the sortie_finished boolean into the rosConnection
struct.
"""
function saveFinished(msg::BoolMsg, rosConnection)
    rosConnection.sortie_finished = msg.data
end

"""
Function to send the next location to Swagbot.
"""
function publishNextLocation(publisher::Publisher{Pose}, new_loc::Location)
    # create Point and Quaternion and put them together
    p = Point(new_loc..., 0)
    # orientation = finalOrientation(pathCost, new_loc)
    orientation = 0
    q = Quaternion(params(QuatRotation(RotZ(orientation)))...)
    publish(publisher, Pose(p, q))
end

end
