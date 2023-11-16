module Samples

using Optim: optimize, ParticleSwarm
using DocStringExtensions: SIGNATURES

using Maps: SampleInput, SampleOutput

"""
Usage: `Sample(x, y)`

Fields:

    - x: the sample input, usually a location and sensor id
    - y: the sample output or observation, a scalar
"""
struct Sample
    x::SampleInput
    y::SampleOutput
end

"""
$(SIGNATURES)

Pulls a ground truth value from a given location and constructs a Sample object
to hold them both.

Inputs:

    - loc: the location to sample
    - sampler: a function that returns ground truth values
    - quantities: (optional) a vector of integers which represent which
      quantities to sample, defaults to all of them

Outputs a vector of Samples containing input x and measurement y
"""
function takeSamples(loc, sampler)
    Y = sampler(loc) # get sample values at location for all quantities
    return [Sample((loc, q), y) for (q, y) in enumerate(Y)]
end

function takeSamples(loc, sampler, quantities)
    return [Sample((loc, q), sampler((loc, q))) for q in quantities]
end

"""
$(SIGNATURES)

The optimization of choosing a best single sample location.

Inputs:

    - occupancy: a map containing upper and lower bounds
    - sampleCost: a function from sample location to cost (x->cost(x))

Returns the sample location, a vector
"""
function selectSampleLocation(sampleCost, lb, ub)
    # in future could optimize for measured quantity as well
    loc0 = (ub .- lb)./2 # I think this value doesn't matter for PSO
    opt = optimize(
        sampleCost,
        loc0,
        ParticleSwarm(; lower=lb, upper=ub, n_particles=20)
    )
    @debug "sample optimizer:" opt
    return opt.minimizer
end

end
