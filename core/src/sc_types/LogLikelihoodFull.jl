
# using AbstractGPs: cov

@doc raw"""
A test of the log likelihood idea but using a weighted sum of all measured
sample values, not just the nearest one:
```math
C(x) = - w_1 \, \frac{1}{\sum_i k(x, x_i)} \sum_i k(x, x_i) *
       \left( \frac{μ(x) - y(x_i)}{σ_n} \right)^2 - w_2 \, \log (σ^2(x))
```
where ``x_i`` is each collected sample location, and ``σ_n`` is the noise.

This function's performance wasn't satisfactory.
"""
struct LogLikelihoodFull <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function LogLikelihoodFull(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    belief_max = nothing

    LogLikelihoodFull(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::LogLikelihoodFull, loc)
    q = 1
    x = (loc, q)
    μ, σ = sc.beliefModel(x) # mean and standard deviation

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    τ = isinf(τ) ? Inf : 0.0

    k = sc.beliefModel.gp.prior.kernel
    K = [k(x, sample.x) for sample in sc.samples]
    K ./= sum(K)
    μ_err_tot = mapreduce(+, enumerate(sc.samples)) do (i, sample)
         K[i] * ((μ - sample.y[1])/sc.beliefModel.θ.σn[q])^2
    end

    # / cov(sc.beliefModel.gp, [x, sample.x])[2,1]

    return (-μ_err_tot, -log(σ^2), τ, 0.0)
end
