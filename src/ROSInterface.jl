#!/usr/bin/env julia

module ROSInterface

using RobotOS: @rosimport, rostypegen
@rosimport std_msgs.msg: Float32
@rosimport geometry_msgs.msg: Pose2D
rostypegen(@__MODULE__)
using .std_msgs.msg
using .geometry_msgs.msg

function callback(msg::Float32Msg, pub_obj::Publisher{Pose2D})
    # new value received

    # check robot location
    # TODO need to import something
    loc = rospy.wait_for_message("/robot_location???", SystemState, timeout=5)

    # create sample
    Sample(loc, msg)

    # run adaptive sampling, publish next location
    # TODO need to get other data accessible here, need to initialize
    sampleCost = SampleCost(region.occupancy, samples, beliefModel, weights)
    x_new = selectSampleLocation(sampleCost, lb, ub)

    next_sample = Pose2D(x_new...)
    publish(pub_obj, next_sample)
end

function main()
    init_node("adaptive_sampling")

    # see if there is some way to pass a Tuple{Vector, Int}
    pub = Publisher{Pose2D}("next_sample", queue_size=10)

    # TODO location
    sub = Subscriber{Pose2D}("/robot_location???", callback, (pub,), queue_size=10)

    # [pub_goalpose, msg_goalpose] = rospublisher('latest_sample','geometry_msgs/Pose');
    # [pub_slowflag, msg_slowflag] = rospublisher('slow_swagbot','std_msgs/Bool');
    # sub_fin = rossubscriber('sortie_finished','DataFormat','struct');
    # [pub_fin, msg_fin] = rospublisher('sortie_finished','std_msgs/Bool');
    # [pub_belief, msg_belief] = rospublisher('belief_fig','sensor_msgs/Image');


    sub = Subscriber{Float32Msg}("/GP_Data/avg_NDVI", callback, (pub,), queue_size=10)

    # values
    sub = Subscriber{Float32Msg}("/GP_Data/avg_NDVI", callback, (pub,), queue_size=10)
    sub = Subscriber{Float32Msg}("/GP_Data/avg_crop_height", callback, (pub,), queue_size=10)
    sub = Subscriber{Float32Msg}("/GP_Data/cover_NDVI", callback, (pub,), queue_size=10)

    spin() # wait for callbacks
end

if !isinteractive()
    main()
end

end
