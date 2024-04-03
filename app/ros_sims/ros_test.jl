#!/usr/bin/env julia
# should be set and run as an executable

# this file is just used for testing ROSInterface.jl

using Pkg
Pkg.activate(Base.source_dir() * "/..")

using PyCall
rospy = pyimport("rospy")
py"""
import rospy
import std_msgs.msg
"""

function loop()
    loop_rate = rospy.Rate(1.0)
    while !rospy.is_shutdown()
        node = "/value1"
        val = py"rospy.wait_for_message($node, std_msgs.msg.Float64, timeout=5)"
        println("Value is $val")
        println("Value is $(val.data)")
    end
end

function main()
    rospy.init_node("rosjl_example")
    loop()
end

if !isinteractive()
    main()
end
