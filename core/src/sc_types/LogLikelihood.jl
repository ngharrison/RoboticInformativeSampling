
@doc raw"""
Idea derived from the log likelihood of query location. Similar to [EIGF](@ref),
the measured value from the nearest sample is used:
```math
C(x) = - w_1 \, \left( \frac{μ(x) - y(x_c)}{σ_n} \right)^2 - w_2 \, \log (σ^2(x))
```
where ``x_c`` is the nearest collected sample location, and ``σ_n`` is the
noise.

This function was seen to work well but hasn't undergone extensive tests. There
is still some question about the theory and the use of noise parameter vs signal
amplitude parameter.
"""
struct LogLikelihood <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function LogLikelihood(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    belief_max = nothing

    LogLikelihood(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::LogLikelihood, loc)
    q = 1
    μ, σ = sc.beliefModel((loc, q)) # mean and standard deviation

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    τ = isinf(τ) ? Inf : 0.0

    closest_sample = argmin(sample -> norm(sample.x[1] - loc), sc.samples)

    μ_err = (μ - closest_sample.y[1])/sc.beliefModel.θ.σn[q]

    return (-μ_err^2, -log(σ^2), τ, 0.0)
end
