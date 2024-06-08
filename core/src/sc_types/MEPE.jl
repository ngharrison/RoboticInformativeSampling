
## This isn't finished, don't work

struct MEPE <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function MEPE(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    # using the max values from the current belief
    locs = [cellToPoint(ci, occupancy) for ci in vec(CartesianIndices(occupancy))]
    belief_max = [maximum(first(beliefModel(tuple.(locs, q)))) for q in quantities]

    MEPE(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::MEPE, loc)
    return (0.0, 0.0, 0.0, 0.0)
end

function (sc::MEPE)(loc)
    # Not implemented yet, would need to look at their paper
    beliefs = sc.beliefModel([(loc, q) for q in sc.quantities]) # means and standard deviations
    μ_norm, σ_norm = mean.(belief ./ sc.belief_max for belief in beliefs)

    d = minimum(norm(sample.x[1] - loc) for sample in sc.samples)

    α = isempty(samples) ? .5 : 0.99*minimum(0.5*(), 1)
    return sc.occupancy(loc) ? Inf : α*vals[1] + (1 - α)*σ^2
end
