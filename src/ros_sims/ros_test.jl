#!/usr/bin/env julia

# using Pkg
# Pkg.activate("../..")

using RobotOS
@rosimport std_msgs.msg: Bool, Float64
@rosimport geometry_msgs.msg: Pose, Point, Quaternion
rostypegen(@__MODULE__)
using .std_msgs.msg
using .geometry_msgs.msg

using PyCall
py"""
import rospy
import std_msgs.msg
"""

function loop()
    loop_rate = Rate(1.0)
    while !is_shutdown()
        node = "/value1"
        val = py"rospy.wait_for_message($node, std_msgs.msg.Float64, timeout=5)"
        println("Value is $val")
        println("Value is $(val.data)")
    end
end

function main()
    init_node("rosjl_example")
    loop()
end

if !isinteractive()
    main()
end
