
"""
$(TYPEDEF)
"""
struct EIGF <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function EIGF(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    belief_max = nothing

    EIGF(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::EIGF, loc)
    μ, σ = sc.beliefModel((loc, 1)) # mean and standard deviation

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    τ = isinf(τ) ? Inf : 0.0

    closest_sample = argmin(sample -> norm(sample.x[1] - loc), sc.samples)

    μ_err = μ - closest_sample.y[1]

    return (-μ_err^2, -σ^2, τ, 0.0)
end
