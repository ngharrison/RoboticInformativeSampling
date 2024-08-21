
# using AbstractGPs: cov

struct LogLikelihoodFull <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function LogLikelihoodFull(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    belief_max = nothing

    LogLikelihoodFull(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::LogLikelihoodFull, loc)
    q = 1
    x = (loc, q)
    μ, σ = sc.beliefModel(x) # mean and standard deviation

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    τ = isinf(τ) ? Inf : 0.0

    k = sc.beliefModel.gp.prior.kernel
    K = [k(x, sample.x) for sample in sc.samples]
    K ./= sum(K)
    μ_err_tot = mapreduce(+, enumerate(sc.samples)) do (i, sample)
         K[i] * ((μ - sample.y[1])/sc.beliefModel.θ.σn[q])^2
    end

    # / cov(sc.beliefModel.gp, [x, sample.x])[2,1]

    return (-μ_err_tot, -log(σ^2), τ, 0.0)
end
