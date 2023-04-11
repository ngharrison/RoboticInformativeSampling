module Samples

using LinearAlgebra
using Optim
using DocStringExtensions

using Paths

export takeSample, selectSampleLocation, SampleCost

"""
Usage: `Sample(x, y)`

Fields:

    - x: the location or index variable, a vector
    - y: the output or observation, a scalar
"""
struct Sample
    x
    y
end

"""
$SIGNATURES

Pulls a ground truth value from a given location and constructs a sample object
to hold them both.

Inputs:

    - x: the location to sample
    - groundTruth: a function that gives ground truth values

Outputs a Sample containing location x and measurement y
"""
function takeSample(x, groundTruth)
    y = groundTruth(x) # get sample value
    return Sample(x, y)
end

"""
$SIGNATURES

The optimization of choosing a best single sample location.

Inputs:

    - region: region data
    - sampleCost: a function from sample location to cost (x->cost(x))
"""
function selectSampleLocation(region, sampleCost)
    x0 = (region.ub .- region.lb)./2 # I think this value doesn't matter for PSO
    opt = optimize(
        sampleCost,
        x0,
        ParticleSwarm(; lower=region.lb, upper=region.ub, n_particles=20)
    )
    return opt.minimizer
end

"""
The cost function used for choosing a new sample location.
"""
struct SampleCost
    region
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
function SampleCost(region, samples, beliefModel, weights)
    pathCost = PathCost(samples[end].x, region.obsMap)
    SampleCost(region, samples, beliefModel, weights, pathCost)
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
    μ, σ = sc.beliefModel(x) # mean and standard deviation
    τ = sc.pathCost(x) # distance to location
    radius = minimum(sc.region.ub .- sc.region.lb)/4
    dists = norm.(getfield.(sc.samples, :x) .- Ref(x))
    P = sum((radius./dists).^3) # proximity to other points
    vals = [-μ, -σ, τ, P]
    return sc.weights'*vals
end

end
