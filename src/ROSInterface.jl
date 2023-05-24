#!/usr/bin/env julia

# ATTENTION: this is not functional yet

module ROSInterface

using RobotOS: std_msgs, geometry_msgs, rostypegen
@rosimport std_msgs.msg: Float32
@rosimport geometry_msgs.msg: Pose, Point, Quaternion
rostypegen(@__MODULE__)
using .std_msgs.msg
using .geometry_msgs.msg

using Rotations: QuatRotation, RotZ

using Environment: Location, Index

"""
Stores information for communicating with Swagbot.
"""
struct ROSConnection
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
    sortie_sub = Subscriber{Float32Msg}("sortie_finished", saveVal,
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

    # wait, TODO how to best implement?
    while !ros_data.sortie_finished
        sleep(1) # choose the publishing rate
    end

    # get values
    values = [wait_for_message(node, Float32Msg, timeout=5) for node in R.sub_nodes]

    return values
end

"""
Returns a single value from the sample location of the chosen quantity.

Currently will be unused.
"""
function (R::ROSConnection)(new_index::Index)
    loc, quantity = new_index
    publishNextLocation(R.publisher, loc)

    # wait, TODO how to best implement?
    while !ros_data.sortie_finished
        sleep(1) # choose the publishing rate
    end

    # get values
    values = wait_for_message(R.sub_nodes[quantity], Float32Msg, timeout=5)

    return values
end

"""
Callback function to save the sortie_finished boolean into the rosConnection
struct.
"""
function saveVal(msg::Float32Msg, rosInterface)
    rosInterface.sortie_finished = msg
end

"""
Function to send the next location to Swagbot.
"""
function publishNextLocation(pub_obj::Publisher{Pose}, new_loc::Location)
    # create Point and Quaternion and put them together
    p = Point(new_loc)
    orientation = finalOrientation(pathCost, new_loc)
    q = Quaternion(QuatRotation(RotZ(orientation)))
    publish(pub_obj, Pose(p, q))
end

end
