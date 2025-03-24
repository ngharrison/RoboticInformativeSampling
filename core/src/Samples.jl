"""
This module contains everything to do with sampling values in the environment.
Its alias types `Location` and `SampleInput` are fundamental pieces for MultiQuantityGPs
as well.

Main public types and functions:
$(EXPORTS)
"""
module Samples

using Optim: optimize, ParticleSwarm
using DocStringExtensions: TYPEDSIGNATURES, TYPEDFIELDS, TYPEDEF, EXPORTS

using GridMaps: GridMap

export Sample, selectSampleLocation, takeSamples, GridMapsSampler, UserSampler,
       Location, SampleInput, getLoc, getQuant, getObs

"""
$(TYPEDEF)
Location of sample
"""
const Location = Vector{Float64}

"""
$(TYPEDEF)
Sample input, the combination of: ([`Location`](@ref), sensor index)
"""
const SampleInput = Tuple{Location, Int}

"""
$(TYPEDEF)
Value of sample measurement, the measurement mean and standard deviation
"""
const SampleOutput = Tuple{Float64, Float64}

"""
Struct to hold the input and output of a sample.

Fields:
$(TYPEDFIELDS)
"""
struct Sample{T}
    "the sample input, usually a location and sensor id"
    x::SampleInput
    "the sample output or observation, a scalar"
    y::T
end

# helpers
getLoc(s::Sample) = s.x[1]
getQuant(s::Sample) = s.x[2]
getObs(s::Sample) = s.y

"""
$(TYPEDSIGNATURES)

Pulls a ground truth value from a given location and constructs a Sample object
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
    return [Sample((loc, q), y) for (q, y) in enumerate(Y)]
end

function takeSamples(loc, sampler, quantities)
    return [Sample((loc, q), sampler((loc, q))) for q in quantities]
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
