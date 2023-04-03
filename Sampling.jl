module Sampling

using Optim
using LinearAlgebra

export takeSample, selectSampleLocation, CostFunction

struct Sample
    x # the location or index variable
    y # the output or observation
end

function takeSample(x, gt)
    y = gt(x) # get sample value
    return Sample(x, y)
end

function selectSampleLocation(region, costFunction)
    x0 = (region.ub .- region.lb)./2 # I think this value doesn't matter for PSO
    opt = optimize(
        costFunction,
        x0,
        ParticleSwarm(; lower=region.lb, upper=region.ub, n_particles=20)
    )
    return opt.minimizer
end

struct CostFunction
    region
    samples
    beliefModel
    weights
end

function (cf::CostFunction)(x)
    # cost to take new sample at location x
    μ, σ = cf.beliefModel(x) # mean and standard deviation
    x_curr = cf.samples[end].x
    τ = pathCost(x_curr, x, cf.region) # distance to location
    radius = minimum(cf.region.ub .- cf.region.lb)/4
    dists = norm.(getfield.(cf.samples, :x) .- Ref(x))
    P = sum((radius./dists).^3) # proximity to other points
    vals = [-μ, -σ, τ, P]
    return cf.weights'*vals
end

function pathCost(x1, x2, region)
    # if either point is within an obstacle, just return infinity
    if any(region.obsMap.([x1, x2]))
        return Inf
    else
        # calculate cost
        # TODO drop in new method, such as A*
        return norm(x2-x1)
    end
end

end
