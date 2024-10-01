
using ..BeliefModels: meanDerivAndVar

"""
$(TYPEDEF)
"""
struct DerivVar <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function DerivVar(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    belief_max = nothing

    DerivVar(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::DerivVar, loc)
    dμ, σ = meanDerivAndVar(sc.beliefModel, (loc, 1)) # mean and standard deviation

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    τ = isinf(τ) ? Inf : 0.0

    return (-dμ^2, -σ^2, τ, 0.0)
end
