
struct DistLogEIGF <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function DistLogEIGF(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    belief_max = nothing

    DistLogEIGF(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::DistLogEIGF, loc)
    μ, σ = sc.beliefModel((loc, 1)) # mean and standard deviation

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    bounds = getBounds(sc.occupancy)
    τ_norm = τ / mean(bounds.upper .- bounds.lower) # normalized

    closest_sample = argmin(sample -> norm(sample.x[1] - loc), sc.samples)

    μ_err = μ - closest_sample.y[1]

    return (-μ_err^2, -log(σ^2), τ_norm^2, 0.0)
end
