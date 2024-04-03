#!/usr/bin/env julia
# should be set and run as an executable

# this file is just used for testing ROSInterface.jl

using Pkg
Pkg.activate(Base.source_dir() * "/..")

using RobotOS
@rosimport std_msgs.msg: Float32
rostypegen()
using .std_msgs.msg

function loop(pubs)
    loop_rate = Rate(2.0)
    while !is_shutdown()
        for pub in pubs
            publish(pub, Float32Msg(rand()))
        end
        rossleep(loop_rate)
    end
end

function main()
    init_node("sample_sim")
    pubs = [Publisher{Float32Msg}("value1", queue_size=1),
            Publisher{Float32Msg}("value2", queue_size=1)]
    loop(pubs)
end

if !isinteractive()
    main()
end
