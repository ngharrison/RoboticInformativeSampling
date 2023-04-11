module Sampling

using LinearAlgebra
using Optim
using Paths

export takeSample, selectSampleLocation, SampleCost

struct Sample
    x # the location or index variable
    y # the output or observation
end

function takeSample(x, groundTruth)
    y = groundTruth(x) # get sample value
    return Sample(x, y)
end

function selectSampleLocation(region, sampleCost)
    x0 = (region.ub .- region.lb)./2 # I think this value doesn't matter for PSO
    opt = optimize(
        sampleCost,
        x0,
        ParticleSwarm(; lower=region.lb, upper=region.ub, n_particles=20)
    )
    return opt.minimizer
end

struct SampleCost
    region
    samples
    beliefModel
    weights
    pathCost
end

function SampleCost(region, samples, beliefModel, weights)
    pathCost = PathCost(samples[end].x, region.obsMap)
    SampleCost(region, samples, beliefModel, weights, pathCost)
end

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
