
using InformativeSampling
using .Maps: Map
using .SampleCosts: EIGFSampleCost
using .Missions: Mission

using InformativeSamplingUtils
using .Visualization: vis

struct UserSampler
    quantities
end

Base.keys(us::UserSampler) = us.quantities

function (us::UserSampler)(x)
    println("At location $x")
    return map(us.quantities) do i
        print("Enter the value for quantity $i: ")
        parse(Float64, readline())
    end
end

function pickMission(; num_samples=30)

    bounds = (lower = [0.0, 0.0], upper = [1.0, 1.0])

    occupancy = Map(zeros(Bool, 100, 100), bounds)

    ## initialize ground truth

    # sampling two quantities each time
    sampler = UserSampler(1:2)

    sampleCostType = EIGFSampleCost

    ## initialize alg values
    weights = (; μ=1, σ=1e2, τ=1, d=0) # others
    start_locs = [[1.0, 0.0]] # starting location

    noise = (0.0, :learned)

    mission = Mission(;
        occupancy,
        sampler,
        num_samples,
        sampleCostType,
        weights,
        start_locs,
        noise
    )

    return mission

end


#* Run

## initialize data for mission
mission = pickMission(num_samples=10)

## run search alg
@time samples, beliefs = mission(
    vis;
    sleep_time=0.0
);
