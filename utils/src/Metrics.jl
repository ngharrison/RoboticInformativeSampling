"""
A module to calculate the metrics from a mission and belief model.

Main public types and functions:
$(EXPORTS)
"""
module Metrics

using Statistics: mean
using LinearAlgebra: norm
using DocStringExtensions: TYPEDSIGNATURES, EXPORTS

using MultiQuantityGPs: MQGP, quantityCorMat
using GridMaps: generateAxes

using InformativeSampling
using .Samples: GridMapsSampler

export calcMetrics

"""
$(TYPEDSIGNATURES)

A collection of methods that calculates metrics from sampling missions.
"""
function calcMetrics end

function calcMetrics(mission, samples, beliefs)
    mission.sampler isa GridMapsSampler ||
        error("don't know how to get a ground truth from that type of sampler")

    M = mission
    axs, points = generateAxes(M.occupancy)

    mae = zeros(length(beliefs), length(M.sampler))
    mse = zeros(length(beliefs), length(M.sampler))
    mu = zeros(length(beliefs), length(M.sampler))
    mb = zeros(length(beliefs), length(M.sampler))
    mxae = zeros(length(beliefs), length(M.sampler))
    mxse = zeros(length(beliefs), length(M.sampler))
    mxu = zeros(length(beliefs), length(M.sampler))
    mxb = zeros(length(beliefs), length(M.sampler))
    cors = Matrix{Any}(undef, (length(beliefs), length(M.sampler)))
    dists = zeros(length(beliefs), length(M.sampler))
    times = zeros(length(beliefs), length(M.sampler))
    for q in eachindex(M.sampler)
        (mae[:,q], mse[:,q], mu[:,q], mb[:,q],
         mxae[:,q], mxse[:,q], mxu[:,q], mxb[:,q],
         cors[:,q], dists[:,q],
         times[:,q]) = calcMetrics(mission, samples, beliefs, times, q, points)
    end

    return (; mae, mse, mu, mb, mxae, mxse, mxu, mxb, cors, dists, times)
end

function calcMetrics(mission, samples, beliefs, times, q)
    mission.sampler isa GridMapsSampler ||
        error("don't know how to get a ground truth from that type of sampler")

    M = mission
    axs, points = generateAxes(M.occupancy)

    return calcMetrics(mission, samples, beliefs, times, q, points)
end

function calcMetrics(mission, samples, beliefs, times, q, points)
    mae = zeros(length(beliefs))
    mse = zeros(length(beliefs))
    mu = zeros(length(beliefs))
    mb = zeros(length(beliefs))
    mxae = zeros(length(beliefs))
    mxse = zeros(length(beliefs))
    mxu = zeros(length(beliefs))
    mxb = zeros(length(beliefs))
    cors = Vector{Any}(undef, (length(beliefs),))
    dists = zeros(length(beliefs))
    for (i, beliefModel) in enumerate(beliefs)
        (mae[i], mse[i], mu[i], mb[i],
         mxae[i], mxse[i], mxu[i], mxb[i],
         cors[i]) = calcMetrics(mission, beliefModel, q, points)
        # distance
        dists[i] = i==1 ? 0.0 : norm(samples[i].x[1] - samples[i-1].x[1])
    end

    return (; mae, mse, mu, mb, mxae, mxse, mxu, mxb, cors, dists, times)
end

function calcMetrics(mission, beliefModel::MQGP, q, points)
    μ, σ = beliefModel(tuple.(vec(points), q))
    true_vals = vec(mission.sampler[q])
    mask = vec(.! mission.occupancy)
    # Mean Absolute Error
    mae = mean(abs.(μ[mask] .- true_vals[mask]))
    # Mean Squared Error
    mse = mean((μ[mask] .- true_vals[mask]).^2)
    # Mean Uncertainty
    mu = mean(σ[vec(mask)])
    # Mean Belief?
    mb = mean(μ[vec(mask)])
    # Max Absolute Error
    mxae = maximum(abs.(μ[mask] .- true_vals[mask]))
    # Max Squared Error
    mxse = maximum((μ[mask] .- true_vals[mask]).^2)
    # Max Uncertainty
    mxu = maximum(σ[vec(mask)])
    # Max Belief?
    mxb = maximum(μ[vec(mask)])
    # Correlations
    cors = quantityCorMat(beliefModel)[:,q]
    return (; mae, mse, mu, mb, mxae, mxse, mxu, mxb, cors)
end

end
