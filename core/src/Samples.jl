"""
This module contains everything to do with sampling values in the environment.

Main public types and functions:
$(EXPORTS)
"""
module Samples

using Optim: optimize, ParticleSwarm
using DocStringExtensions: TYPEDSIGNATURES, TYPEDFIELDS, TYPEDEF, EXPORTS

using MultiQuantityGPs: MQSample, Location, MQSampleInput

export selectSampleLocation, takeSamples, GridMapsSampler, UserSampler

"""
$(TYPEDSIGNATURES)

Pulls a ground truth value from a given location and constructs a MQSample object
to hold them both.

Inputs:
- `loc`: the location to sample
- `sampler`: a function that returns ground truth values
- `quantities`: (optional) a vector of integers which represent which
  quantities to sample, defaults to all of them

Outputs a vector of Samples containing input x and measurement y
"""
function takeSamples(loc, sampler)
    Y = sampler(loc) # get sample values at location for all quantities
    return [MQSample(((loc, q), y)) for (q, y) in enumerate(Y)]
end

function takeSamples(loc, sampler, quantities)
    return [MQSample(((loc, q), sampler((loc, q)))) for q in quantities]
end

"""
$(TYPEDSIGNATURES)

The optimization of choosing a best single sample location.

Inputs:
- `sampleCost`: a function from sample location to cost (x->cost(x))
- `bounds`: map lower and upper bounds

Returns the sample location, a vector
"""
function selectSampleLocation(sampleCost, bounds)
    # in future could optimize for measured quantity as well
    loc0 = @. (bounds.upper - bounds.lower)/2 + bounds.lower
    opt = optimize(
        sampleCost,
        loc0,
        ParticleSwarm(; bounds.lower, bounds.upper, n_particles=40)
    )
    @debug "sample optimizer:" opt
    return opt.minimizer
end


### Samplers ###

include("samplers/GridMapsSampler.jl")

include("samplers/UserSampler.jl")

end
