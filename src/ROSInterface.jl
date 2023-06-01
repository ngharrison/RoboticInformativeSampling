#!/usr/bin/env julia

# ATTENTION: this is not functional yet

module ROSInterface

using RobotOS
@rosimport std_msgs.msg: Bool, Float64
@rosimport geometry_msgs.msg: Pose, Point, Quaternion
rostypegen(@__MODULE__)
using .std_msgs.msg
using .geometry_msgs.msg

using PyCall
rospy = pyimport("rospy")
std_msgs = pyimport("std_msgs.msg")

using Rotations: QuatRotation, RotZ, params

using Environment: Location, Index

"""
Stores information for communicating with Swagbot.
"""
mutable struct ROSConnection
    sub_nodes
    publisher
    sortie_finished
end

"""
Constructor.
"""
function ROSConnection(sub_nodes)
    # initialize this node with its name
    init_node("adaptive_sampling")

    # this will pass the full goal pose, no quantity id
    publisher = Publisher{Pose}("latest_sample", queue_size=10)

    # create connection object
    rosConnection = ROSConnection(sub_nodes, publisher, false)

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
    publishNextLocation(R.publisher, new_loc)

    # wait, check each second
    while !ros_data.sortie_finished
        rossleep(1)
    end

    # get values
    values = [rospy.wait_for_message(node, std_msgs.Float64, timeout=5) for node in R.sub_nodes]

    return values
end

"""
Returns a single value from the sample location of the chosen quantity.

Currently will be unused.
"""
function (R::ROSConnection)(new_index::Index)
    loc, quantity = new_index
    publishNextLocation(R.publisher, loc)

    # wait, check each second
    while !ros_data.sortie_finished
        rossleep(1)
    end

    # get values
    values = rospy.wait_for_message(R.sub_nodes[quantity], std_msgs.Float64, timeout=5)

    return values
end

"""
Callback function to save the sortie_finished boolean into the rosConnection
struct.
"""
function saveFinished(msg::BoolMsg, rosInterface)
    rosInterface.sortie_finished = msg
end

"""
Function to send the next location to Swagbot.
"""
function publishNextLocation(pub_obj::Publisher{Pose}, new_loc::Location)
    # create Point and Quaternion and put them together
    p = Point([new_loc; 0]...)
    # orientation = finalOrientation(pathCost, new_loc)
    orientation = 0
    q = Quaternion(params(QuatRotation(RotZ(orientation)))...)
    publish(publisher, Pose(p, q))
end

end
