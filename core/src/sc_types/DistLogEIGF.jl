
@doc raw"""
A variation on [EIGF](@ref) that takes the logarithm of the variance and adds a
distance cost term that is normalized by the average of the region dimensions:
```math
C(x) = - w_1 \, (μ(x) - y(x_c))^2 - w_2 \, \log(σ^2(x)) +
       w_3 \, β \, \frac{τ(x)}{\left\lVert \boldsymbol{\ell}_d \right\rVert_1}
```
where ``β`` is a parameter to delay the distance effect until a few samples have
been taken.
"""
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
    n_scale = 2/(1 + exp(1 - length(sc.samples))) - 1

    closest_sample = argmin(sample -> norm(sample.x[1] - loc), sc.samples)

    μ_err = μ - closest_sample.y[1]

    return (-μ_err^2, -log(σ^2), n_scale*τ_norm^2, 0.0)
end
