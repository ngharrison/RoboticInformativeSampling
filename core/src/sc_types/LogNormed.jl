
"""
$(TYPEDEF)
"""
struct LogNormed <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function LogNormed(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    # using the max values from the current belief
    locs = [cellToPoint(ci, occupancy) for ci in vec(CartesianIndices(occupancy))]
    belief_max = [maximum(first(beliefModel(tuple.(locs, q)))) for q in quantities]

    LogNormed(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::LogNormed, loc)
    beliefs = sc.beliefModel([(loc, q) for q in sc.quantities]) # means and standard deviations
    μ_norm, σ_norm = mean.(belief ./ sc.belief_max for belief in beliefs) # normed and averaged

    τ_norm = sc.occupancy(loc) ? Inf : 0.0
    # τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    # bounds = getBounds(sc.occupancy)
    # τ_norm = τ / mean(bounds.upper .- bounds.lower) # normalized

    return (-μ_norm, -log(σ_norm), τ_norm, 0.0)
end
