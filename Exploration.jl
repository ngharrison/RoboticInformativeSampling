module Exploration

using Sampling
using BeliefModels
using Visualization

export explore

function explore(region, x_start, weights; num_samples=20, show_visuals=false, sleep_time=0)
    # adaptive sampling
    samples = []
    beliefModel = BeliefModel(nothing)
    costFunction = CostFunction(region, samples, beliefModel, weights)
    x_new = x_start
    num_samples=20
    for i in 1:num_samples
        println("Sample number $i")

        # new sample
        if beliefModel.gp !== nothing # prior belief exists
            x_new = selectSampleLocation(region, costFunction)

            μ, σ = beliefModel(x_new)
            @debug "Sample location: $x_new"
            @debug "Location values: $([μ, σ])"
            @debug "Location costs: $([μ*weights[1], σ*weights[2]])"

            sleep(sleep_time)
        end

        sample = takeSample(x_new, region.gtMap)
        push!(samples, sample)

        # new belief
        update!(beliefModel, region, samples)
        if show_visuals
            display(visualize(region.gtMap, beliefModel, samples, costFunction, region))
        end
    end
    println("Mission complete")
    return samples, beliefModel
end

end
