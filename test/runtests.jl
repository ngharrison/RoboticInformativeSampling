using Test

using AdaptiveSampling
using .Paths: PathCost
using .Maps: Map, res, pointToCell, cellToPoint, generateAxes

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

    @test pathCost(CartesianIndex(7,9)) ≈ 7 + 4√2 # retry
end

@testset "Maps" begin
    n = 3
    m = reshape(1:n^2, n,n)
    map = Map(m, [1,1], 2 .* [n,n] .- 1)

    @test all(res(map) .== [2.0, 2.0])

    a = [CartesianIndex(i,j) for i in 1:n, j in 1:n]
    b = cellToPoint.([CartesianIndex(i,j) for i in 1:n, j in 1:n], Ref(map))
    c = pointToCell.(b, Ref(map))
    @test a == c

    axs, points = generateAxes(map)
    @test b == points
end
