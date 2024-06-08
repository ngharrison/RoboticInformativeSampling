module SampleCosts

using LinearAlgebra: norm
using Statistics: mean
using DocStringExtensions: TYPEDSIGNATURES

using ..Maps: pointToCell, cellToPoint, res, getBounds
using ..Paths: PathCost

export SampleCost, values, BasicSampleCost,
       NormedSampleCost, MIPTSampleCost, EIGFSampleCost,
       DistScaledEIGFSampleCost, DistEIGFSampleCost

abstract type SampleCost end

"""
$(TYPEDSIGNATURES)

Cost to take a new sample at a location. This is a fallback method that
calculates a simple linear combination of all the values of a SampleCost.

Has the form:
``\\mathrm{cost} = - w_{1} μ - w_{2} σ + w_{3} τ + w_{4} D``
"""
function (sc::SampleCost)(loc)
    # sc.occupancy(loc) && return Inf
    vals = values(sc, loc)
    return sum(w*v for (w,v) in zip(sc.weights, vals))
end

"""
A basic cost function used for choosing a new sample location.
"""
struct BasicSampleCost <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    pathCost
end

"""
$(TYPEDSIGNATURES)

A pathCost is constructed automatically from the other arguments.

This object can then be called to get the cost of sampling at a location:
sampleCost(x)
"""
function BasicSampleCost(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))
    BasicSampleCost(occupancy, samples, beliefModel,
                    quantities, weights, pathCost)
end

"""
$(TYPEDSIGNATURES)

Returns the values to be used to calculate the sample cost (belief mean,
standard deviation, travel distance, sample proximity).
"""
function values(sc::BasicSampleCost, loc)
    beliefs = sc.beliefModel([(loc, q) for q in sc.quantities]) # means and standard deviations
    μ_ave, σ_ave = mean.(beliefs)

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location

    bounds = getBounds(sc.occupancy)
    radius = minimum(bounds.upper .- bounds.lower)/4
    dists = norm.(sample.x[1] - loc for sample in sc.samples)
    P = sum((radius./dists).^3) # proximity to other points

    return (-μ_ave, -σ_ave, τ, P)
end

struct NormedSampleCost <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function NormedSampleCost(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    # using the max values from the current belief
    locs = [cellToPoint(ci, occupancy) for ci in vec(CartesianIndices(occupancy))]
    belief_max = [maximum(first(beliefModel(tuple.(locs, q)))) for q in quantities]

    NormedSampleCost(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::NormedSampleCost, loc)
    beliefs = sc.beliefModel([(loc, q) for q in sc.quantities]) # means and standard deviations
    μ_norm, σ_norm = mean.(belief ./ sc.belief_max for belief in beliefs) # normed and averaged

    τ_norm = sc.occupancy(loc) ? Inf : 0.0
    # τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    # bounds = getBounds(sc.occupancy)
    # τ_norm = τ / mean(bounds.upper .- bounds.lower) # normalized

    return (-μ_norm, -log(σ_norm), τ_norm, 0.0)
end

struct MIPTSampleCost <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function MIPTSampleCost(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    # using the max values from the current belief
    locs = [cellToPoint(ci, occupancy) for ci in vec(CartesianIndices(occupancy))]
    belief_max = [maximum(first(beliefModel(tuple.(locs, q)))) for q in quantities]

    MIPTSampleCost(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::MIPTSampleCost, loc)
    # τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    # bounds = getBounds(sc.occupancy)
    # τ_norm = τ / mean(bounds.upper .- bounds.lower) # normalized
    τ_norm = sc.occupancy(loc) ? Inf : 0.0
    d = minimum(norm(sample.x[1] - loc) for sample in sc.samples)
    return (0.0, 0.0, τ_norm, -d)
end

struct EIGFSampleCost <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function EIGFSampleCost(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    belief_max = nothing

    EIGFSampleCost(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::EIGFSampleCost, loc)
    μ, σ = sc.beliefModel((loc, 1)) # mean and standard deviation

    # τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    # bounds = getBounds(sc.occupancy)
    # τ_norm = τ / mean(bounds.upper .- bounds.lower) # normalized
    τ_norm = sc.occupancy(loc) ? Inf : 0.0

    closest_sample = argmin(sample -> norm(sample.x[1] - loc), sc.samples)

    μ_err = μ - closest_sample.y[1]

    return (-μ_err^2, -σ^2, τ_norm, 0.0)
end

struct DistScaledEIGFSampleCost <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function DistScaledEIGFSampleCost(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    belief_max = nothing

    DistScaledEIGFSampleCost(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::DistScaledEIGFSampleCost, loc)
    μ, σ = sc.beliefModel((loc, 1)) # mean and standard deviation

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    bounds = getBounds(sc.occupancy)
    τ_norm = τ / mean(bounds.upper .- bounds.lower) # normalized

    closest_sample = argmin(sample -> norm(sample.x[1] - loc), sc.samples)

    μ_err = μ - closest_sample.y[1]

    d = (τ_norm == Inf ? Inf : 0.0)

    # gradually delay distance scaling
    n_scale = 2/(1 + exp(1 - length(sc.samples))) - 1
    d_scale = 1/(1 + n_scale*τ_norm^2)
    d_scale = isnan(d_scale) ? 1.0 : d_scale # prevent 0*Inf=NaN

    return (-μ_err^2*d_scale, -σ^2*d_scale, d, 0.0)
end

struct DistEIGFSampleCost <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function DistEIGFSampleCost(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    belief_max = nothing

    DistEIGFSampleCost(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::DistEIGFSampleCost, loc)
    μ, σ = sc.beliefModel((loc, 1)) # mean and standard deviation

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    bounds = getBounds(sc.occupancy)
    τ_norm = τ / mean(bounds.upper .- bounds.lower) # normalized

    closest_sample = argmin(sample -> norm(sample.x[1] - loc), sc.samples)

    μ_err = μ - closest_sample.y

    return (-μ_err^2, -log(σ^2), τ_norm^2, 0.0)
end

## These aren't finished, don't work

struct MEPESampleCost <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function MEPESampleCost(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    # using the max values from the current belief
    locs = [cellToPoint(ci, occupancy) for ci in vec(CartesianIndices(occupancy))]
    belief_max = [maximum(first(beliefModel(tuple.(locs, q)))) for q in quantities]

    MEPESampleCost(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::MEPESampleCost, loc)
    return (0.0, 0.0, 0.0, 0.0)
end

function (sc::MEPESampleCost)(loc)
    # Not implemented yet, would need to look at their paper
    beliefs = sc.beliefModel([(loc, q) for q in sc.quantities]) # means and standard deviations
    μ_norm, σ_norm = mean.(belief ./ sc.belief_max for belief in beliefs)

    d = minimum(norm(sample.x[1] - loc) for sample in sc.samples)

    α = isempty(samples) ? .5 : 0.99*minimum(0.5*(), 1)
    return sc.occupancy(loc) ? Inf : α*vals[1] + (1 - α)*σ^2
end

end
