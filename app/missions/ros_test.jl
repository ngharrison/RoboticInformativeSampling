#!/usr/bin/env julia

using Pkg
Pkg.activate(Base.source_dir() * "/..")

using Logging: global_logger, ConsoleLogger, Info, Debug

using GridMaps: GridMap

using InformativeSampling
using .SampleCosts: EIGF
using .Missions: Mission

# this requires a working rospy installation
include(dirname(Base.active_project()) * "/ros/ROSInterface.jl")
using .ROSInterface: ROSSampler

function rosMission(; num_samples=4)

    # the topics that will be listened to for measurements
    data_topics = [
        ("value1", "value2")
    ]

    done_topic = "sortie_finished"
    pub_topic = "latest_sample"

    sampler = ROSSampler(data_topics, done_topic, pub_topic)

    bounds = (lower = [1.0, 1.0], upper = [9.0, 9.0])

    occupancy = GridMap(zeros(Bool, 100, 100), bounds)

    sampleCostType = EIGF

    ## initialize alg values
    weights = (; μ=1, σ=5e3, τ=1, d=1) # mean, std, dist, prox
    start_locs = [[1.0, 1.0]]

    mission = Mission(;
        occupancy,
        sampler,
        num_samples,
        sampleCostType,
        weights,
        start_locs
    )

    return mission
end

#* Run

# set the logging level: Info or Debug
global_logger(ConsoleLogger(stderr, Debug))

using InformativeSamplingUtils
using .Visualization: vis
using .DataIO: save

## initialize data for mission
mission = rosMission()

ros_version = readchomp(`rosversion -d`)
python_exe = "python" * (ros_version == "noetic" ? "3" : "")

# start other scripts
cmds = (`$(python_exe) $(Base.source_dir())/../ros/sample_sim.py`,
        `$(python_exe) $(Base.source_dir())/../ros/sortie_sim.py`)

procs = run.(cmds; wait=false)

try
    ## run search alg
    @time samples, beliefs = mission(
        # vis;
        sleep_time=0.0
    )
finally
    # kill other scripts
    kill.(procs)
end
