module Metrics

using Statistics: mean
using BeliefModels: BeliefModel, outputCorMat

function calcMetrics(mission, beliefs)
    M = mission
    axs = range.(M.occupancy.lb, M.occupancy.ub, size(M.occupancy))
    points = collect.(Iterators.product(axs...))

    mae = zeros(length(beliefs), length(M.sampler))
    mu = zeros(length(beliefs), length(M.sampler))
    mb = zeros(length(beliefs), length(M.sampler))
    cors = Matrix{Any}(undef, (length(beliefs), length(M.sampler)))
    for (q, map) in enumerate(M.sampler)
        mae[:,q], mu[:,q], mb[:,q], cors[:,q] = calcMetrics(mission, beliefs, q, points)
    end

    return (; mae, mu, mb, cors)
end

function calcMetrics(mission, beliefs, q)
    M = mission
    axs = range.(M.occupancy.lb, M.occupancy.ub, size(M.occupancy))
    points = collect.(Iterators.product(axs...))

    mae = zeros(length(beliefs))
    mu = zeros(length(beliefs))
    mb = zeros(length(beliefs))
    cors = Vector{Any}(undef, (length(beliefs),))
    for (i, beliefModel) in enumerate(beliefs)
        mae[i], mu[i], mb[i], cors[i] = calcMetrics(mission, beliefModel, q, points)
    end

    return (; mae, mu, mb, cors)
end

function calcMetrics(mission, beliefs, q, points)
    mae = zeros(length(beliefs))
    mu = zeros(length(beliefs))
    mb = zeros(length(beliefs))
    cors = Vector{Any}(undef, (length(beliefs),))
    for (i, beliefModel) in enumerate(beliefs)
        mae[i], mu[i], mb[i], cors[i] = calcMetrics(mission, beliefModel, q, points)
    end

    return (; mae, mu, mb, cors)
end

function calcMetrics(mission, beliefModel::BeliefModel, q, points)
    μ, σ = beliefModel(tuple.(vec(points), q))
    true_vals = vec(mission.sampler[q])
    mask = vec(.! mission.occupancy)
    # Mean Absolute Error
    mae = mean(abs.(μ[mask] .- true_vals[mask]))
    # Mean Uncertainty
    mu = mean(σ[vec(mask)])
    # Mean Belief?
    mb = mean(μ[vec(mask)])
    # Maxes?
    # Correlations
    cors = outputCorMat(beliefModel)[:,q]
    return (; mae, mu, mb, cors)
end

end
