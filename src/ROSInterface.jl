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
    publisher
end

"""
Constructor.
"""
function ROSConnection(sub_nodes)
    # initialize this node with its name
    init_node("adaptive_sampling")

    # this will pass the full goal pose, no quantity id
    publisher = Publisher{Pose}("latest_sample", queue_size=1, latch=true)

    # create connection object
    rosConnection = ROSConnection(sub_nodes, publisher)

    return rosConnection
end

# give it a length and indices
Base.eachindex(R::ROSConnection) = eachindex(R.sub_nodes)
Base.length(R::ROSConnection) = length(R.sub_nodes)

"""
Returns a vector of values from the sample location.
"""
function (R::ROSConnection)(new_loc::Location)
    publishNextLocation(R.publisher, new_loc)

    rospy = pyimport("rospy")
    std_msg = pyimport("std_msgs.msg")

    rospy.wait_for_message("sortie_finished", std_msg.Bool) # a message means finished

    # get values
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
    publishNextLocation(R.publisher, loc)

    rospy = pyimport("rospy")
    std_msg = pyimport("std_msgs.msg")

    rospy.wait_for_message("sortie_finished", std_msg.Bool) # a message means finished

    # get values
    values = rospy.wait_for_message(R.sub_nodes[quantity], std_msg.Float64, timeout=5)

    return values
end

"""
Function to send the next location to Swagbot.
"""
function publishNextLocation(publisher::Publisher{Pose}, new_loc::Location)
    # create Point and Quaternion and put them together
    p = Point(new_loc..., 0)
    # TODO use real orientation
    # orientation = finalOrientation(pathCost, new_loc)
    orientation = 0
    q = Quaternion(params(QuatRotation(RotZ(orientation)))...)
    publish(publisher, Pose(p, q))
end

end
