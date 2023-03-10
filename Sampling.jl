module Sampling

using Optim
using BeliefModel
using LinearAlgebra

export takeSample, selectSampleLocation, createCostFunc

struct Sample
    x
    y
end

function takeSample(x, gt)
    y = gt(x) # get sample value
    return Sample(x, y)
end

function selectSampleLocation(region, samples, belief_model, weights)
    x0 = (region.ub .- region.lb)./2 # I think this value doesn't matter for PSO
    opt = optimize(
        createCostFunc(region, samples, belief_model, weights),
        x0,
        ParticleSwarm(; lower=region.lb, upper=region.ub, n_particles=20)
    )
    return opt.minimizer
end

pathCost(x1, x2) = norm(x2-x1)

function createCostFunc(region, samples, belief_model, weights)
    # return cost function
    x -> begin
        # cost to take new sample at location x given current location x_curr
        x_curr = samples[end].x
        μ, σ = only.(getBelief([x], belief_model)) # mean and standard deviation
        τ = pathCost(x_curr, x) # time to location
        radius = minimum(region.ub .- region.lb)/4
        dists = norm.(getfield.(samples, :x) .- Ref(x))
        P = sum((radius./dists).^3) # proximity to other points
        vals = [-μ, -σ, τ, P]
        return weights'*vals
    end
end

end
