
@doc raw"""
Combines the average mean value, average standard deviation, travel distance,
and proximity as terms:
```math
C(x) = - w_1 \, μ_{\mathrm{ave}}(x) - w_2 \, σ_{\mathrm{ave}}(x) +
       w_3 \, τ(x) + w_4 \, P(x)
```
where ``P(x) = \sum_i(\frac{\min(\boldsymbol{\ell}_d)}{4 \, \mathrm{dist}_i})^3``.
Averages are performed over all quantities.
"""
struct DistProx <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    pathCost
end

function DistProx(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))
    DistProx(occupancy, samples, beliefModel,
                    quantities, weights, pathCost)
end

function values(sc::DistProx, loc)
    beliefs = sc.beliefModel([(loc, q) for q in sc.quantities]) # means and standard deviations
    μ_ave, σ_ave = mean.(beliefs)

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location

    bounds = getBounds(sc.occupancy)
    radius = minimum(bounds.upper .- bounds.lower)/4
    dists = norm.(sample.x[1] - loc for sample in sc.samples)
    P = sum((radius./dists).^3) # proximity to other points

    return (-μ_ave, -σ_ave, τ, P)
end
