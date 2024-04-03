#!/usr/bin/env julia
# should be set and run as an executable

# this file is just used for testing ROSInterface.jl

using Pkg
Pkg.activate(Base.source_dir() * "/..")

using RobotOS
@rosimport std_msgs.msg: Bool
@rosimport geometry_msgs.msg: PoseStamped
rostypegen(@__MODULE__)
using .std_msgs.msg
using .geometry_msgs.msg

function callback(pose::PoseStamped, pub::Publisher{BoolMsg})
    println("Pose received!")
    println(pose)
    rossleep(1)
    publish(pub, BoolMsg(true)) # sortie finished
end

function main()
    init_node("sortie_sim")
    pub = Publisher{BoolMsg}("sortie_finished", queue_size=1)
    sub = Subscriber{PoseStamped}("latest_sample", callback, (pub,), queue_size=1)
    spin()
end

if !isinteractive()
    main()
end
