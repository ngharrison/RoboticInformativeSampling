
using Test

using InformativeSampling
using .Paths: PathCost, finalOrientation, getPath
using .Maps: Map, res, pointToCell, cellToPoint, generateAxes
using .Missions: Mission
using .Samples: MapsSampler
using .SampleCosts: EIGFSampleCost

@testset "Paths" begin
    occupancy = zeros(Bool, 10, 10) # open area
    start = CartesianIndex(2,3)
    pathCost = PathCost(start, occupancy, (1, 1))

    @test pathCost(start) == 0 # zero
    @test pathCost(start + CartesianIndex(1,0)) == 1 # single cell down
    @test pathCost(start + CartesianIndex(-1,0)) == 1 # single cell up
    @test pathCost(start + CartesianIndex(0,1)) == 1 # single cell right
    @test pathCost(start + CartesianIndex(0,-1)) == 1 # single cell left

    @test pathCost(start + CartesianIndex(1,1)) ≈ √2 # diagonal
    a, b = 2, 4
    @test pathCost(start + CartesianIndex(a,b)) ≈ abs(b-a) + min(a,b)*√2 # straight and diagonal

    @test pathCost(start + CartesianIndex(0,7)) ≈ 7 # to edge

    a, b = 8, 7
    @test pathCost(start + CartesianIndex(a,b)) ≈ abs(b-a) + min(a,b)*√2 # to corner

    # out of bounds
    @test_throws BoundsError pathCost(start + CartesianIndex(-2,0))

    a, b = 2, 4
    r1, r2 = .24, 1.1 # different resolution
    pathCost = PathCost(start, occupancy, (r1,r2))
    @test pathCost(start + CartesianIndex(a,b)) ≈ (b-a)*r2 + a√(r1^2+r2^2) # straight and diagonal

    # obstacle
    occupancy = zeros(Bool, 10, 10)
    obstacle = CartesianIndex.([
        tuple.(2, 3:5)...,
        tuple.(3, 3:5)...,
        tuple.(4, 4:6)...,
        tuple.(5, 5:7)...,
        tuple.(6, 4:6)...,
        tuple.(7, 3:5)...,
        tuple.(8, 3:5)...,
    ])
    occupancy[obstacle] .= 1

    start = CartesianIndex(4,3)
    pathCost = PathCost(start, occupancy, (1, 1))
    @test pathCost(CartesianIndex(7,9)) ≈ 7 + 4√2

    @test pathCost(CartesianIndex(7,9)) ≈ 7 + 4√2 # redo

    path = [
        CartesianIndex(4, 3),
        CartesianIndex(5, 3),
        CartesianIndex(6, 3),
        CartesianIndex(7, 2),
        CartesianIndex(8, 2),
        CartesianIndex(9, 3),
        CartesianIndex(9, 4),
        CartesianIndex(9, 5),
        CartesianIndex(9, 6),
        CartesianIndex(9, 7),
        CartesianIndex(8, 8),
        CartesianIndex(7, 9)
    ]

    @test getPath(pathCost, CartesianIndex(7,9)) == path
    @test finalOrientation(pathCost, CartesianIndex(7,9)) ≈ 3π / 4
end

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

    sampleCostType = EIGFSampleCost

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
