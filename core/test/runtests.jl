
using Test

using InformativeSampling
using .Maps: Map, res, pointToCell, cellToPoint, generateAxes
using .Missions: Mission
using .Samples: MapsSampler
using .SampleCosts: EIGF

@testset "Maps" begin
    n = 3
    m = reshape(1:n^2, n,n)
    map = Map(m, (lower=[1,1], upper=2 .* [n,n] .- 1))

    @test all(res(map) .== [2.0, 2.0])

    a = [CartesianIndex(i,j) for i in 1:n, j in 1:n]
    b = cellToPoint.([CartesianIndex(i,j) for i in 1:n, j in 1:n], Ref(map))
    c = pointToCell.(b, Ref(map))
    @test a == c

    axs, points = generateAxes(map)
    @test b == points

    # inside and boundaries
    @test pointToCell([1,1], map) == CartesianIndex(1,1)
    @test pointToCell([1.99,1.99], map) == CartesianIndex(1,1)
    @test pointToCell([2.01,2.01], map) == CartesianIndex(2,2)
    @test pointToCell([3,3], map) == CartesianIndex(2,2)
    @test pointToCell([3.99,3.99], map) == CartesianIndex(2,2)
    @test pointToCell([4.01,4.01], map) == CartesianIndex(3,3)
    @test pointToCell([5,5], map) == CartesianIndex(3,3)
end

# Missions: standard mission run remains the same (will break a lot)
@testset "Missions" begin
    # set up mission
    bounds = (lower = [0.0, 0.0], upper = [1.0, 1.0])
    dims = (100, 100)
    occupancy = Map(zeros(Bool, dims), bounds)

    axs = range.(bounds..., dims)
    mat = [i*sin(10j) for i in axs[1], j in axs[2]]
    map0 = Map(mat, bounds)
    sampler = MapsSampler(map0)

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
