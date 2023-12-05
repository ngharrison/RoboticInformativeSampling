using AdaptiveSampling: Maps, SampleCosts, ROSInterface, Missions

using .Maps: Map
using .SampleCosts: EIGFSampleCost
using .Missions: Mission

# this requires a working rospy installation
using .ROSInterface: ROSConnection

function rosMission(; num_samples=4)

    # the topics that will be listened to for measurements
    sub_topics = [
        # Crop height avg and std in frame (excluding wheels)
        ("value1", "value2")
    ]

    sampler = ROSConnection(sub_topics)

    lb = [1.0, 1.0]; ub = [9.0, 9.0]

    occupancy = Map(zeros(Bool, 100, 100), lb, ub)

    sampleCostType = EIGFSampleCost

    ## initialize alg values
    weights = (; μ=1, σ=5e3, τ=1, d=1) # mean, std, dist, prox
    start_locs = [[1.0, 1.0]]s

    return Mission(; occupancy,
                   sampler,
                   num_samples,
                   sampleCostType,
                   weights,
                   start_locs)
end
