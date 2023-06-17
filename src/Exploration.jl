module Exploration

using DocStringExtensions: SIGNATURES

using Samples: Sample, selectSampleLocation, takeSamples
using SampleCosts: SampleCost, BasicSampleCost, NormedSampleCost, values
using BeliefModels: BeliefModel
using Visualization: visualize

"""
$SIGNATURES

The main function that runs the adaptive sampling routine. For each iteration, a
sample location is selected, a sample is collected, the belief model is updated,
and visuals are possibly shown. The run finishes when the designated number of
samples is collected.

Inputs:

    - md: the mission data struct
    - visuals: true or false to cause map plots to be shown or not
    - sleep_time: the amount of time to wait after each iteration, useful for
      visualizations

Outputs:

    - samples: the new samples collected
    - beliefModel: the probabilistic representation of the quantities being
      searched for
"""
function explore(md; visuals=false, sleep_time=0)
    md.occupancy(md.start_loc) && error("start location is within obstacle")

    # initialize
    lb, ub = md.occupancy.lb, md.occupancy.ub
    new_loc = md.start_loc
    quantities = eachindex(md.groundTruth) # all current available quantities
    samples = copy(md.samples)

    beliefModel = nothing
    if !isempty(samples)
        beliefModel = BeliefModel(samples, md.prior_samples, lb, ub)
    end
    sampleCost = nothing

    println("Mission started")
    println()

    for i in 1:md.num_samples
        println("Sample number $i")

        # new sample indices
        if beliefModel !== nothing # prior belief exists
            sampleCost = NormedSampleCost(md, samples, beliefModel, quantities)
            new_loc = selectSampleLocation(sampleCost, lb, ub)
            # @show values(sampleCost, new_loc)
        end

        # sample all quantities
        new_samples = takeSamples(new_loc, md.groundTruth)
        append!(samples, new_samples)

        # new belief
        beliefModel = BeliefModel(samples, md.prior_samples, lb, ub)

        # visualization
        if visuals
            display(visualize(md, beliefModel, sampleCost, samples, quantity=1))
        end
        # println("Output correlations: $(round.(outputCorMat(beliefModel), digits=3))")
        sleep(sleep_time)
    end

    println()
    println("Mission complete")
    return samples, beliefModel
end

end
