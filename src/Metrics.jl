module Metrics

using Statistics: mean

function calcMetrics(mission, samples, beliefs)
    M = mission
    axs = range.(M.occupancy.lb, M.occupancy.ub, size(M.occupancy))
    dims = Tuple(length.(axs))
    points = collect.(Iterators.product(axs...))

    mae = zeros(length(beliefs), length(M.sampler))
    mu = zeros(length(beliefs), length(M.sampler))
    mb = zeros(length(beliefs), length(M.sampler))
    for (i, beliefModel) in enumerate(beliefs)
        for (q, map) in enumerate(M.sampler)
            μ, σ = beliefModel(tuple.(vec(points), q))
            pred_map = reshape(μ, dims)
            true_vals = M.sampler[q]
            mask = .! M.occupancy
            # Mean Absolute Error
            mae[i,q] = mean(abs.(pred_map[mask] .- true_vals[mask]))
            # Mean Uncertainty
            mu[i,q] = mean(σ[vec(mask)])
            # Mean Belief?
            mb[i,q] = mean(μ[vec(mask)])
            # Maxes?
        end
    end

    return (; mae, mu, mb)
end

end
