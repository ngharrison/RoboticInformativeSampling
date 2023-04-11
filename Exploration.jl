module Exploration

using DocStringExtensions

using Samples
using BeliefModels

export explore

"""
$SIGNATURES

The main function that runs the adaptive sampling routine. For each iteration, a
sample location is selected, a sample is collected, the belief model is updated,
and visuals are possibly shown. The run finishes when the designated number of
samples is collected.

Inputs:

    - region: the search region
    - x_start: the starting location
    - weights: weights for picking the next sample location (optional)
    - num_samples: the number of samples to collect in one run
    - visualize: the function to use to see plots of the algorithm progress, if
      nothing, no plots will be shown
    - sleep_time: the amount of time to wait after each iteration, useful for
      visualizations

Outputs:

    - samples: the samples collected
    - beliefModel: the probabilistic representation of the quantity being
      searched for
"""
function explore(region, x_start, weights; num_samples=20, visualize=nothing, sleep_time=0)
    region.obsMap(x_start) && error("start location is within obstacle")

    samples = []
    beliefModel = nothing
    x_new = x_start

    for i in 1:num_samples
        println("Sample number $i")

        # new sample
        if beliefModel !== nothing # prior belief exists
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
