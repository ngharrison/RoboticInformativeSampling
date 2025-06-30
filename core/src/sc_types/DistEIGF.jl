
@doc raw"""
Augments [EIGF](@ref) with a normalized travel distance term:
```math
C(x) = - w_1 \, (μ(x) - y(x_c))^2 - w_2 \, σ^2(x) + w_3 \, γ \, τ^2(x)
```
where ``β`` is a parameter to delay the distance effect until a few samples have
been taken.
"""
struct DistEIGF <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function DistEIGF(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    belief_max = nothing

    DistEIGF(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::DistEIGF, loc)
    μ, σ = sc.beliefModel((loc, 1)) # mean and standard deviation

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    bounds = getBounds(sc.occupancy)
    τ_norm = τ / mean(bounds.upper .- bounds.lower) # normalized

    closest_sample = argmin(sample -> norm(sample.x[1] - loc), sc.samples)

    μ_err = μ - closest_sample.y[1]

    n_scale = 2/(1 + exp(1 - length(sc.samples))) - 1

    return (-μ_err^2, -σ^2, n_scale*τ_norm^2, 0.0)
end
