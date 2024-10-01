
"""
$(TYPEDEF)
"""
struct OnlyVar <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function OnlyVar(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    belief_max = nothing

    OnlyVar(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::OnlyVar, loc)
    _, σ = sc.beliefModel((loc, 1)) # mean and standard deviation

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    τ = isinf(τ) ? Inf : 0.0

    return (0.0, -σ^2, τ, 0.0)
end
