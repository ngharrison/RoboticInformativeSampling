#!/usr/bin/env julia

module ROSInterface

using RobotOS
@rosimport geometry_msgs.msg: Pose, Point, Quaternion
rostypegen(@__MODULE__)
using .geometry_msgs.msg

using PyCall: pyimport

using Rotations: QuatRotation, RotZ, params

using Samples: Location, SampleInput

export ROSConnection

"""
Stores information for communicating with Swagbot.

Objects of this type can be used as samplers in missions.
"""
struct ROSConnection
    sub_topics
    publisher
end

"""
Constructor.
"""
function ROSConnection(sub_topics)
    # initialize this node with its name
    init_node("adaptive_sampling")

    # this will pass the full goal pose, no quantity id
    publisher = Publisher{Pose}("latest_sample", queue_size=1, latch=true)

    # create connection object
    rosConnection = ROSConnection(sub_topics, publisher)

    return rosConnection
end

# give it a length and indices
Base.eachindex(R::ROSConnection) = eachindex(R.sub_topics)
Base.length(R::ROSConnection) = length(R.sub_topics)

"""
Returns a vector of values from the sample location.
"""
function (R::ROSConnection)(new_loc::Location)
    publishNextLocation(R.publisher, new_loc)

    rospy = pyimport("rospy")
    std_msg = pyimport("std_msgs.msg")

    rospy.wait_for_message("sortie_finished", std_msg.Bool) # a message means finished
    @debug "received sortie finished"

    # get values
    values = [rospy.wait_for_message(node, std_msg.Float32, timeout=20).data
              for node in R.sub_topics]
    @debug "received values:" values

    return values
end

"""
Returns a single value from the sample location of the chosen quantity.

Currently will be unused.
"""
function (R::ROSConnection)(new_index::SampleInput)
    loc, quantity = new_index
    publishNextLocation(R.publisher, loc)

    rospy = pyimport("rospy")
    std_msg = pyimport("std_msgs.msg")

    rospy.wait_for_message("sortie_finished", std_msg.Bool) # a message means finished
    @debug "received sortie finished"

    # get values
    values = rospy.wait_for_message(R.sub_topics[quantity], std_msg.Float32, timeout=20)
    @debug "received values:" values

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
    pose = Pose(p, q)
    publish(publisher, pose)
    @debug "published next location:" pose
end

end
