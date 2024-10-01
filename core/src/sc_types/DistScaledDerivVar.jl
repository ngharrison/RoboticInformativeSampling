
using ..BeliefModels: meanDerivAndVar

"""
$(TYPEDEF)
"""
struct DistScaledDerivVar <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function DistScaledDerivVar(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    belief_max = nothing

    DistScaledDerivVar(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::DistScaledDerivVar, loc)
    dμ, σ = meanDerivAndVar(sc.beliefModel, (loc, 1)) # mean and standard deviation

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    bounds = getBounds(sc.occupancy)
    τ_norm = τ / mean(bounds.upper .- bounds.lower) # normalized

    d = (τ_norm == Inf ? Inf : 0.0)

    # gradually delay distance scaling
    n_scale = 2/(1 + exp(1 - length(sc.samples))) - 1
    d_scale = 1/(1 + n_scale*τ_norm^2)
    d_scale = isnan(d_scale) ? 1.0 : d_scale # prevent 0*Inf=NaN

    return (-dμ^2*d_scale, -σ^2*d_scale, d, 0.0)
end
