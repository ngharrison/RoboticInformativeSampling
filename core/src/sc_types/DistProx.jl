
"""
$(TYPEDEF)

A basic cost function used for choosing a new sample location.
"""
struct DistProx <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    pathCost
end

"""
$(TYPEDSIGNATURES)

A pathCost is constructed automatically from the other arguments.

This object can then be called to get the cost of sampling at a location:
sampleCost(x)
"""
function DistProx(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))
    DistProx(occupancy, samples, beliefModel,
                    quantities, weights, pathCost)
end

"""
$(TYPEDSIGNATURES)

Returns the values to be used to calculate the sample cost (belief mean,
standard deviation, travel distance, sample proximity).
"""
function values(sc::DistProx, loc)
    beliefs = sc.beliefModel([(loc, q) for q in sc.quantities]) # means and standard deviations
    μ_ave, σ_ave = mean.(beliefs)

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location

    bounds = getBounds(sc.occupancy)
    radius = minimum(bounds.upper .- bounds.lower)/4
    dists = norm.(sample.x[1] - loc for sample in sc.samples)
    P = sum((radius./dists).^3) # proximity to other points

    return (-μ_ave, -σ_ave, τ, P)
end
