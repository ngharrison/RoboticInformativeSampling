module Metrics

using Statistics: mean

using InformativeSampling
using .Maps: generateAxes
using .BeliefModels: BeliefModel, outputCorMat
using .Samples: MapsSampler

export calcMetrics

function calcMetrics(mission, beliefs)
    mission.sampler isa MapsSampler ||
        error("don't know how to get a ground truth from that type of sampler")

    M = mission
    axs, points = generateAxes(M.occupancy)

    mae = zeros(length(beliefs), length(M.sampler))
    mu = zeros(length(beliefs), length(M.sampler))
    mb = zeros(length(beliefs), length(M.sampler))
    mxae = zeros(length(beliefs), length(M.sampler))
    mxu = zeros(length(beliefs), length(M.sampler))
    mxb = zeros(length(beliefs), length(M.sampler))
    cors = Matrix{Any}(undef, (length(beliefs), length(M.sampler)))
    for (q, map) in enumerate(M.sampler)
        (mae[:,q], mu[:,q], mb[:,q],
         mxae[:,q], mxu[:,q], mxb[:,q], cors[:,q]) = calcMetrics(mission, beliefs, q, points)
    end

    return (; mae, mu, mb, mxae, mxu, mxb, cors)
end

function calcMetrics(mission, beliefs, q)
    mission.sampler isa MapsSampler ||
        error("don't know how to get a ground truth from that type of sampler")

    M = mission
    axs, points = generateAxes(M.occupancy)

    mae = zeros(length(beliefs))
    mu = zeros(length(beliefs))
    mb = zeros(length(beliefs))
    mxae = zeros(length(beliefs))
    mxu = zeros(length(beliefs))
    mxb = zeros(length(beliefs))
    cors = Vector{Any}(undef, (length(beliefs),))
    for (i, beliefModel) in enumerate(beliefs)
        (mae[i], mu[i], mb[i],
         mxae[i], mxu[i], mxb[i], cors[i]) = calcMetrics(mission, beliefModel, q, points)
    end

    return (; mae, mu, mb, mxae, mxu, mxb, cors)
end

function calcMetrics(mission, beliefs, q, points)
    mae = zeros(length(beliefs))
    mu = zeros(length(beliefs))
    mb = zeros(length(beliefs))
    mxae = zeros(length(beliefs))
    mxu = zeros(length(beliefs))
    mxb = zeros(length(beliefs))
    cors = Vector{Any}(undef, (length(beliefs),))
    for (i, beliefModel) in enumerate(beliefs)
        (mae[i], mu[i], mb[i],
         mxae[i], mxu[i], mxb[i], cors[i]) = calcMetrics(mission, beliefModel, q, points)
    end

    return (; mae, mu, mb, mxae, mxu, mxb, cors)
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
    # Max Absolute Error
    mxae = maximum(abs.(μ[mask] .- true_vals[mask]))
    # Max Uncertainty
    mxu = maximum(σ[vec(mask)])
    # Max Belief?
    mxb = maximum(μ[vec(mask)])
    # Correlations
    cors = outputCorMat(beliefModel)[:,q]
    return (; mae, mu, mb, mxae, mxu, mxb, cors)
end

end
