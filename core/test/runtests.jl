
using Test

using GridMaps: GridMap

using InformativeSampling
using .Missions: Mission
using .Samples: GridMapsSampler
using .SampleCosts: EIGF

# Missions: standard mission run doesn't error
@testset "Missions" begin
    # set up mission
    bounds = (lower = [0.0, 0.0], upper = [1.0, 1.0])
    dims = (100, 100)
    occupancy = GridMap(zeros(Bool, dims), bounds)

    axs = range.(bounds..., dims)
    mat = [i*sin(10j) for i in axs[1], j in axs[2]]
    map0 = GridMap(mat, bounds)
    sampler = GridMapsSampler(map0)

    sampleCostType = EIGF

    ## initialize alg values
    weights = (; μ=1, σ=1e2, τ=1, d=0) # others
    start_locs = [] # starting locations
    num_samples = 5

    mission = Mission(;
        occupancy,
        sampler,
        num_samples,
        sampleCostType,
        weights,
    )

    @test_nowarn mission()
end
