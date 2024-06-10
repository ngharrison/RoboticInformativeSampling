#!/usr/bin/env julia

module ROSInterface

using PyCall: pyimport, PyNULL
using DocStringExtensions: TYPEDSIGNATURES, TYPEDFIELDS, FUNCTIONNAME

using ..Samples: Location, SampleInput

export ROSConnection

const rospy = PyNULL()
const std_msg = PyNULL()
const geo_msg = PyNULL()

function __init__()
    copy!(rospy, pyimport("rospy"))
    copy!(std_msg, pyimport("std_msgs.msg"))
    copy!(geo_msg, pyimport("geometry_msgs.msg"))
end

"""
A struct that stores information for communicating with Swagbot.

Objects of this type can be used as samplers in missions, meaning they can be
called with a SampleInput to return its value. This object also has a length,
which is the length of the number of its subscriptions and can be iterated over
to get the name of each one.

Fields:
$(TYPEDFIELDS)
"""
struct ROSConnection{T<:Union{String, NTuple{2, String}}}
    "vector of topic names that will be subscribed to to receive measurements"
    data_topics::Vector{T}
    "topic name that publishes a message to signify the traveling is done"
    done_topic
    "the publisher topic name"
    pub_topic
    "the publisher topic object, created automatically from a given name"
    publisher
end

"""
$(TYPEDSIGNATURES)

Creating a $(FUNCTIONNAME) object requires a vector of topics to subscribe to
for measurement data, essentially the list of sensors onboard the robot to
listen to. Each element of this list should be a 2-tuple of topics that will
transmit the value and error for each sensor. This constructor initializes a ros
node and sets up a publisher to pub_topic.
"""
function ROSConnection(data_topics, done_topic, pub_topic)
    # initialize this node with its name
    rospy.init_node("informative_sampling")

    # this will pass the full goal pose, no quantity id
    publisher = rospy.Publisher(pub_topic, geo_msg.PoseStamped, queue_size=1, latch=true)

    # create connection object
    rosConnection = ROSConnection(data_topics, done_topic, pub_topic, publisher)

    return rosConnection
end

# give it a length and indices
Base.eachindex(R::ROSConnection) = eachindex(R.data_topics)
Base.length(R::ROSConnection) = length(R.data_topics)

"""
```julia
function (R::ROSConnection{String})(new_loc::Location)
```

Returns a vector of values from the sample location, one for each sensor
measurement available. It does this by first publishing the next location to
sample. Once the location is sampled, it calls out to each topic in sequence and
waits for its message.

# Examples
```julia
data_topics = [
    "/value1",
    "/value2"
]

done_topic = "sortie_finished"
pub_topic = "latest_sample"

sampler = ROSConnection(data_topics, done_topic, pub_topic)

location = [.1, .3]
[value1, value2] = sampler(location)
```

```julia
function (R::ROSConnection{NTuple{2, String}})(new_loc::Location)
```

Returns a vector of (value, error) pairs from the sample location, one for each
sensor measurement available. It does this by first publishing the next location
to sample. Once the location is sampled, it calls out to each topic in sequence
and waits for its message.

# Examples
```julia
data_topics = [
    ("/value1", "/error1"),
    ("/value2", "/error2")
]

done_topic = "sortie_finished"
pub_topic = "latest_sample"

sampler = ROSConnection(data_topics, done_topic, pub_topic)

location = [.1, .3]
[(value1, error1), (value2, error2)] = sampler(location)
```
"""
function (R::ROSConnection{String})(new_loc::Location)
    publishNextLocation(R.publisher, new_loc)

    rospy.wait_for_message(R.done_topic, std_msg.Bool) # a message means finished
    @debug "received traveling done"

    # get a list of values
    observations = [
        rospy.wait_for_message(val_topic, std_msg.Float32, timeout=20).data
        for val_topic in R.data_topics]

    @debug "received values:" observations

    return observations
end

function (R::ROSConnection{NTuple{2, String}})(new_loc::Location)
    publishNextLocation(R.publisher, new_loc)

    rospy.wait_for_message(R.done_topic, std_msg.Bool) # a message means finished
    @debug "received traveling done"

    # get values, creates a list of tuples of (value, error)
    observations = [
        (rospy.wait_for_message(val_topic, std_msg.Float32, timeout=20).data,
            rospy.wait_for_message(err_topic, std_msg.Float32, timeout=20).data)
        for (val_topic, err_topic) in R.data_topics]

    @debug "received values:" observations

    return observations
end

"""
```julia
function (R::ROSConnection{String})(new_index::SampleInput)
```

Returns a single value from the sample location of the chosen quantity.  It does
this by first publishing the next location to sample. Once the location is
sampled, it calls out to each topic in sequence and waits for its message.

Currently will be unused.

```julia
function (R::ROSConnection{NTuple{2, String}})(new_index::SampleInput)
```

Returns a single value and its error from the sample location of the chosen
quantity.  It does this by first publishing the next location to sample. Once
the location is sampled, it calls out to each topic in sequence and waits for
its message.

Currently will be unused.
"""
function (R::ROSConnection{String})(new_index::SampleInput)
    loc, quantity = new_index
    publishNextLocation(R.publisher, loc)

    rospy = pyimport("rospy")
    std_msg = pyimport("std_msgs.msg")

    rospy.wait_for_message(R.done_topic, std_msg.Bool) # a message means finished
    @debug "received traveling done"

    # get value
    val_topic = R.data_topics[quantity]
    value = rospy.wait_for_message(val_topic, std_msg.Float32, timeout=20).data
    @debug "received value:" value

    return value
end

function (R::ROSConnection{NTuple{2, String}})(new_index::SampleInput)
    loc, quantity = new_index
    publishNextLocation(R.publisher, loc)

    rospy = pyimport("rospy")
    std_msg = pyimport("std_msgs.msg")

    rospy.wait_for_message(R.done_topic, std_msg.Bool) # a message means finished
    @debug "received traveling done"

    # get value
    (val_topic, err_topic) = R.data_topics[quantity]
    value = rospy.wait_for_message(val_topic, std_msg.Float32, timeout=20).data
    error = rospy.wait_for_message(err_topic, std_msg.Float32, timeout=20).data
    @debug "received value:" value, error

    return value, error
end

"""
$(TYPEDSIGNATURES)

Internal function used to send the next location to Swagbot.
"""
function publishNextLocation(publisher, new_loc::Location)
    # create Point and Quaternion and put them together
    p = geo_msg.Point(new_loc..., 0)
    q = geo_msg.Quaternion(1, 0, 0, 0) # orientation is ignored
    pose = geo_msg.Pose(p, q)
    poseStamped = geo_msg.PoseStamped()
    poseStamped.header.stamp = rospy.Time.now()
    poseStamped.header.frame_id = "utm"
    poseStamped.pose = pose
    publisher.publish(poseStamped)
    @debug "published next location:" pose
end

end
