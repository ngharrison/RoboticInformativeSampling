module Samples

using LinearAlgebra
using Optim
using DocStringExtensions

using Paths

export Sample, takeSample, selectSampleLocation, SampleCost

"""
Usage: `Sample(x, y)`

Fields:

    - x: a tuple of the location and the index of measured quantity
      can be either a SO for a single output
      or a MO for multiple outputs
    - y: the output or observation, a scalar
"""
struct Sample
    x::Tuple{Vector{Float64}, Int}
    y::Float64
end

"""
$SIGNATURES

Pulls a ground truth value from a given location and constructs a sample object
to hold them both.

Inputs:

    - x: the location to sample
    - region: region data, including a function for ground truth values

Outputs a Sample containing location x and measurement y
"""
function takeSample(x, region)
    y = region.gtMap(x[1]) # get value at sample location
    return Sample(x, y)
end

"""
$SIGNATURES

The optimization of choosing a best single sample location.

Inputs:

    - occMap: a map containing upper and lower bounds
    - sampleCost: a function from sample location to cost (x->cost(x))
"""
function selectSampleLocation(sampleCost, lb, ub)
    x0 = (ub .- lb)./2 # I think this value doesn't matter for PSO
    opt = optimize(
        sampleCost,
        x0,
        ParticleSwarm(; lower=lb, upper=ub, n_particles=20)
    )
    return opt.minimizer
end

"""
The cost function used for choosing a new sample location.
"""
struct SampleCost
    occMap
    samples
    beliefModel
    weights
    pathCost
end

"""
$SIGNATURES

A pathCost is constructed automatically from the other arguments.

This object can then be called to get the cost of sampling at a location:
sampleCost(x)
"""
function SampleCost(occMap, samples, beliefModel, weights)
    pathCost = PathCost(samples[end].x[1], occMap) # just looking at locations
    SampleCost(occMap, samples, beliefModel, weights, pathCost)
end

"""
Cost to take a new sample at location x.
Combines belief mean and standard deviation, travel distance,
and sample proximity.

Has the form:
cost = - w1 μ - w2 σ + w3 τ + w4 D
"""
function (sc::SampleCost)(x)
    # cost to take new sample at location x
    μ, σ = sc.beliefModel((x, 1)) # mean and standard deviation
    τ = sc.pathCost(x) # distance to location
    radius = minimum(sc.occMap.ub .- sc.occMap.lb)/4
    dists = norm.(first.(getfield.(sc.samples, :x)) .- Ref(x))
    P = sum((radius./dists).^3) # proximity to other points
    vals = [-μ, -σ, τ, P]
    return sc.weights'*vals
end

end
