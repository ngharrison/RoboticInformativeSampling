#!/usr/bin/env julia

module ROSInterface

using RobotOS
@rosimport std_msgs.msg: Bool, Float32
@rosimport geometry_msgs.msg: Pose, Point, Quaternion
rostypegen(@__MODULE__)
using .std_msgs.msg
using .geometry_msgs.msg

using PyCall

using Rotations: QuatRotation, RotZ, params

using Samples: Location, SampleInput

export ROSConnection

"""
Stores information for communicating with Swagbot.

Objects of this type can be used as samplers in missions.
"""
struct ROSConnection
    sub_topics
    sample_pub
    sortie_pub
    sortie_finished
end

function callback(val::BoolMsg)
    println("Sortie received!")
    println(val)
    R.sortie_finished = val
    #rossleep(1)
end


"""
Constructor.
"""
function ROSConnection(sub_topics)
    # initialize this node with its name
    init_node("adaptive_sampling")

    # this will pass the full goal pose, no quantity id
    sample_pub = Publisher{Pose}("latest_sample", queue_size=1, latch=true)

    sortie_pub = Publisher{BoolMsg}("sortie_finished", queue_size=1)
    sortie_sub = Subscriber{BoolMsg}("sortie_finished", callback, queue_size=1)

    sortie_finished = false
    # create connection object
    rosConnection = ROSConnection(sub_topics, sample_pub, sortie_pub, sortie_finished)

    return rosConnection
end

# give it a length and indices
Base.eachindex(R::ROSConnection) = eachindex(R.sub_topics)
Base.length(R::ROSConnection) = length(R.sub_topics)

"""
Returns a vector of values from the sample location.
"""
function (R::ROSConnection)(new_loc::Location)
    publishNextLocation(R.sample_pub, new_loc)

    rospy = pyimport("rospy")
    std_msg = pyimport("std_msgs.msg")
    
    var = false
    #while var == false
    var = rospy.wait_for_message("sortie_finished", std_msg.Bool).data
    println("var is $var")
    #end # a message means finished
    println("Spinning for sortie finished")
    #while R.sortie_finished == false spinOnce() end
    println("publishing sortie false")

    #publish(R.sortie_pub, BoolMsg(false))

    # get values
    values = [rospy.wait_for_message(node, std_msg.Float32, timeout=20).data
              for node in R.sub_topics]

    return values
end

"""
Returns a single value from the sample location of the chosen quantity.

Currently will be unused.
"""
function (R::ROSConnection)(new_index::SampleInput)
    loc, quantity = new_index
    publishNextLocation(R.sample_pub, loc)

    rospy = pyimport("rospy")
    std_msg = pyimport("std_msgs.msg")

    rospy.wait_for_message("sortie_finished", std_msg.Bool) # a message means finished
    publish(R.sortie_pub, false)

    # get values
    values = rospy.wait_for_message(R.sub_topics[quantity], std_msg.Float32, timeout=20)

    return values
end

"""
Function to send the next location to Swagbot.
"""
function publishNextLocation(sample_pub::Publisher{Pose}, new_loc::Location)
    # create Point and Quaternion and put them together
    p = Point(new_loc..., 0)
    # TODO use real orientation
    # orientation = finalOrientation(pathCost, new_loc)
    orientation = 0
    q = Quaternion(params(QuatRotation(RotZ(orientation)))...)
    println("Published next sample location")
    publish(sample_pub, Pose(p, q))
end

end
