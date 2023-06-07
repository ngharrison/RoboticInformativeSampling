#!/usr/bin/env julia

# this file is just used for testing ROSInterface.jl

using Pkg
Pkg.activate("..")

using RobotOS
@rosimport std_msgs.msg: Float64
rostypegen()
using .std_msgs.msg

function loop(pubs)
    loop_rate = Rate(2.0)
    while !is_shutdown()
        for pub in pubs
            publish(pub, Float64Msg(rand()))
        end
        rossleep(loop_rate)
    end
end

function main()
    init_node("sample_sim")
    pubs = [Publisher{Float64Msg}("value1", queue_size=1),
            Publisher{Float64Msg}("value2", queue_size=1)]
    loop(pubs)
end

if !isinteractive()
    main()
end
