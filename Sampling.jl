module Sampling

using Optim
using BeliefModel
using LinearAlgebra

export takeSample, selectSampleLocation, createCostFunc

function takeSample(x, gt)
    y = gt(x) # get sample value
    return (; x, y)
end

function selectSampleLocation(region, samples, belief_model, weights)
    lower = [region.x1.lb, region.x2.lb]
    upper = [region.x1.ub, region.x2.ub]
    x0 = (upper .- lower)./2 # I think this doesn't matter for PSO
    opt = optimize(
        createCostFunc(region, samples, belief_model, weights),
        x0,
        ParticleSwarm(; lower, upper, n_particles=50)
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
        lower = [region.x1.lb, region.x2.lb]
        upper = [region.x1.ub, region.x2.ub]
        radius = minimum(upper .- lower)/4
        dists = norm.(getfield.(samples, :x) .- Ref(x))
        P = sum((radius./dists).^3) # proximity to other points
        vals = [-μ, -σ, τ, P]
        return weights'*vals
    end
end

end
