module Samples

using Optim: optimize, ParticleSwarm
using DocStringExtensions: SIGNATURES

using Environment: Index

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
    Y = groundTruth(loc) # get sample values at location for all quantities
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

end
