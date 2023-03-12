module Exploration

using Sampling
using BeliefModel
using Visualization

export explore

function explore(region, x_start, weights, gt; num_samples=20, show_visuals=false, sleep_time=0)
    # adaptive sampling
    samples = []
    belief_model = nothing
    x_new = x_start
    num_samples=20
    for i in 1:num_samples
        println("Sample number $i")

        # new sample
        if belief_model !== nothing # prior belief exists
            x_new = selectSampleLocation(region, samples, belief_model, weights)

            μ, σ = getBelief(x_new, belief_model)
            @debug "Sample location: $x_new"
            @debug "Location values: $([μ, σ])"
            @debug "Location costs: $([μ*weights[1], σ*weights[2]])"

            sleep(sleep_time)
        end

        sample = takeSample(x_new, gt)
        push!(samples, sample)

        # new belief
        belief_model = generateBeliefModel(region, samples)
        if show_visuals
            display(visualize(gt, belief_model, samples, weights, region))
        end
    end
    println("Mission complete")
    return samples, belief_model
end

end
