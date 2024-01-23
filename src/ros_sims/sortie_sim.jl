#!/usr/bin/env julia
# should be run as `julia --project <script_name>` from within a project directory

# this file is just used for testing ROSInterface.jl

using RobotOS
@rosimport std_msgs.msg: Bool
@rosimport geometry_msgs.msg: Pose
rostypegen(@__MODULE__)
using .std_msgs.msg
using .geometry_msgs.msg

function callback(pose::Pose, pub::Publisher{BoolMsg})
    println("Pose received!")
    println(pose)
    rossleep(5)
    println("Sortie published")
    publish(pub, BoolMsg(true)) # sortie finished
end

function main()
    init_node("sortie_sim")
    pub = Publisher{BoolMsg}("sortie_finished", queue_size=1)
    sub = Subscriber{Pose}("latest_sample", callback, (pub,), queue_size=1)
    spin()
end

if !isinteractive()
    main()
end
