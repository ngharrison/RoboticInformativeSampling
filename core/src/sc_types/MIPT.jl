
"""
$(TYPEDEF)
"""
struct MIPT <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function MIPT(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    # using the max values from the current belief
    locs = [cellToPoint(ci, occupancy) for ci in vec(CartesianIndices(occupancy))]
    belief_max = [maximum(first(beliefModel(tuple.(locs, q)))) for q in quantities]

    MIPT(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::MIPT, loc)
    # τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    # bounds = getBounds(sc.occupancy)
    # τ_norm = τ / mean(bounds.upper .- bounds.lower) # normalized
    τ_norm = sc.occupancy(loc) ? Inf : 0.0
    d = minimum(norm(sample.x[1] - loc) for sample in sc.samples)
    return (0.0, 0.0, τ_norm, -d)
end
