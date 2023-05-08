module Exploration

using DocStringExtensions

using Samples
using BeliefModels
using Visualization

export explore

"""
$SIGNATURES

The main function that runs the adaptive sampling routine. For each iteration, a
sample location is selected, a sample is collected, the belief model is updated,
and visuals are possibly shown. The run finishes when the designated number of
samples is collected.

Inputs:

    - region: the search region
    - start_loc: the starting location
    - weights: weights for picking the next sample location
    - num_samples: the number of samples to collect in one run (default 20)
    - prior_samples: any samples taken previously (default empty)
    - visuals: true or false to cause map plots to be shown or not
    - sleep_time: the amount of time to wait after each iteration, useful for
      visualizations

Outputs:

    - samples: the new samples collected
    - beliefModel: the probabilistic representation of the quantities being
      searched for
"""
function explore(region, start_loc, weights;
                 num_samples=20,
                 prior_samples=Sample[],
                 quantities=eachindex(region.groundTruth),
                 visuals=false,
                 sleep_time=0)
    region.occupancy(start_loc) && error("start location is within obstacle")

    lb, ub = region.occupancy.lb, region.occupancy.ub
    samples = empty(prior_samples)
    beliefModel = nothing
    sample_indices = [(start_loc, q) for q in quantities]

    for i in 1:num_samples
        println("Sample number $i")

        # new sample
        if beliefModel !== nothing # prior belief exists
            sampleCost = SampleCost(region.occupancy, samples, beliefModel, quantities, weights)
            new_loc = selectSampleLocation(sampleCost, lb, ub)
            sample_indices = [(new_loc, q) for q in quantities] # sample all quantities
        end

        new_samples = takeSample.(sample_indices, region.groundTruth)
        append!(samples, new_samples)

        # new belief
        beliefModel = generateBeliefModel(samples, prior_samples, lb, ub)

        # visualization
        if visuals
            display(visualize(beliefModel, region, samples, 1))
        end
        # @show correlations(beliefModel)
        sleep(sleep_time)
    end

    println("Mission complete")
    return samples, beliefModel
end

end
