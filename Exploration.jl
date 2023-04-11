module Exploration

using Samples
using BeliefModels

export explore

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
            sampleCost = SampleCost(region, samples, beliefModel, weights)
            x_new = selectSampleLocation(region, sampleCost)
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

end
