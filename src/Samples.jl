module Samples

using LinearAlgebra: norm
using Optim: optimize, ParticleSwarm
using DocStringExtensions: SIGNATURES

using Environment: Index, pointToCell, res
using Paths: PathCost

"""
Usage: `Sample(x, y)`

Fields:

    - x: a tuple of the location and the index of measured quantity
         also called the sample index
    - y: the output or observation, a scalar
"""
struct Sample
    x::Index
    y::Float64
end

"""
$SIGNATURES

Pulls a ground truth value from a given location and constructs a Sample object
to hold them both.

Inputs:

    - loc: the location to sample
    - groundTruth: a function that returns ground truth values
    - quantities: (optional) a vector of integers which represent which
      quantities to sample, defaults to all of them

Outputs a vector of Samples containing index x and measurement y
"""
function takeSamples(loc, groundTruth)
    Y = groundTruth(loc) # get values of samples with location and quantity
    return [Sample((loc, q), y) for (q, y) in enumerate(Y)]
end

function takeSamples(loc, groundTruth, quantities)
    return [Sample((loc, q), groundTruth((loc, q))) for q in quantities]
end

"""
$SIGNATURES

The optimization of choosing a best single sample location.

Inputs:

    - occupancy: a map containing upper and lower bounds
    - sampleCost: a function from sample location to cost (x->cost(x))
"""
function selectSampleLocation(sampleCost, lb, ub)
    # in future could optimize for measured quantity as well
    loc0 = (ub .- lb)./2 # I think this value doesn't matter for PSO
    opt = optimize(
        sampleCost,
        loc0,
        ParticleSwarm(; lower=lb, upper=ub, n_particles=20)
    )
    return opt.minimizer
end

"""
The cost function used for choosing a new sample location.
"""
struct SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    pathCost
end

"""
$SIGNATURES

A pathCost is constructed automatically from the other arguments.

This object can then be called to get the cost of sampling at a location:
sampleCost(x)
"""
function SampleCost(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))
    SampleCost(occupancy, samples, beliefModel, quantities, weights, pathCost)
end

"""
Cost to take a new sample at a location.
Combines belief mean and standard deviation, travel distance,
and sample proximity.

Has the form:
cost = - w1 μ - w2 σ + w3 τ + w4 D
"""
function (sc::SampleCost)(loc)

    beliefs = sc.beliefModel([(loc, q) for q in sc.quantities]) # means and standard deviations
    # TODO probably need to normalize before adding,
    # but requires generating belief over entire region
    # or using the max so far
    μ_tot, σ_tot = sum.(beliefs)

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location

    radius = minimum(sc.occupancy.ub .- sc.occupancy.lb)/4
    dists = norm.(first.(getfield.(sc.samples, :x)) .- Ref(loc))
    P = sum((radius./dists).^3) # proximity to other points

    vals = [-μ_tot, -σ_tot, τ, P]
    return sc.weights'*vals
end

end
