module Exploration

using LinearAlgebra
using Sampling
using BeliefModels
using Paths
using Plots

export explore, CostFunction

# adaptive sampling
function explore(region, x_start, weights; num_samples=20, visualize=nothing, sleep_time=0)
    region.obsMap(x_start) && error("start location is within obstacle")

    samples = []
    beliefModel = nothing
    x_new = x_start

    for i in 1:num_samples
        println("Sample number $i")

        if beliefModel !== nothing # prior belief exists
            # new sample
            costFunction = CostFunction(region, samples, beliefModel, weights)
            x_new = selectSampleLocation(region, costFunction)
        end

        sample = takeSample(x_new, region.gtMap)
        push!(samples, sample)

        # new belief
        beliefModel = generateBeliefModel(samples, region)

        # visualization
        if visualize |> !isnothing
            display(visualize(beliefModel, region.gtMap, samples, region))
        end
        sleep(sleep_time)
    end

    println("Mission complete")
    return samples, beliefModel
end

struct CostFunction
    region
    samples
    beliefModel
    weights
    pathCost
end

function CostFunction(region, samples, beliefModel, weights)
    pathCost = PathCost(samples[end].x, region.obsMap)
    CostFunction(region, samples, beliefModel, weights, pathCost)
end

function (cf::CostFunction)(x)
    # cost to take new sample at location x
    μ, σ = cf.beliefModel(x) # mean and standard deviation
    τ = cf.pathCost(x) # distance to location
    radius = minimum(cf.region.ub .- cf.region.lb)/4
    dists = norm.(getfield.(cf.samples, :x) .- Ref(x))
    P = sum((radius./dists).^3) # proximity to other points
    vals = [-μ, -σ, τ, P]
    return cf.weights'*vals
end

end
