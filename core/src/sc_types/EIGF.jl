
@doc raw"""
Expected Informativeness for Global Fit (EIGF). This function is adapted from
[^Lam] through added weights to choose the balance between exploration and
exploitation.
It has the form:
```math
C(x) = - w_1 \, (μ(x) - y(x_c))^2 - w_2 \, σ^2(x)
```
where ``x_c`` is the nearest collected sample location.

[^Lam]: Lam, C Q (2008) Sequential adaptive designs in computer experiments for response surface model fit (Doctoral dissertation). The Ohio State University.
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
