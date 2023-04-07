module Sampling

using Optim

export takeSample, selectSampleLocation

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

end
