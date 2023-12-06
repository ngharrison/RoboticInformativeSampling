using Test

using AdaptiveSampling: Paths
using .Paths: PathCost

@testset "Paths tests" begin
    occupancy = zeros(Bool, 10, 10)
    start = CartesianIndex(2,3)
    pathCost = PathCost(start, occupancy, (1, 1))

    @test pathCost(CartesianIndex(2,3)) == 0
    @test pathCost(CartesianIndex(2,4)) == 1
    @test pathCost(CartesianIndex(3,3)) == 1
    @test pathCost(CartesianIndex(3,4)) ≈ √2
end
