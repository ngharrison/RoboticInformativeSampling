module Missions

using Random: seed!
using DocStringExtensions: TYPEDSIGNATURES, TYPEDFIELDS

using ..Maps: randomPoint, bounds
using ..Samples: Sample, selectSampleLocation, takeSamples
using ..BeliefModels: BeliefModel, outputCorMat
using ..Kernels: multiKernel
using ..SampleCosts: values

export Mission, replay

"""
Fields:
$(TYPEDFIELDS)

Defined as a keyword struct, so all arguments are passed in as keywords:
```julia
mission = Mission(; occupancy,
                  sampler,
                  num_samples,
                  sampleCostType,
                  weights,
                  start_locs,
                  prior_samples)
```
"""
@kwdef struct Mission
    "an occupancy map, true in cells that are occupied"
    occupancy
    "a function that returns a measurement value for any input"
    sampler
    "the number of samples to collect in one run"
    num_samples
    "a constructor for the function that returns the (negated) value of taking a sample"
    sampleCostType
    "weights for picking the next sample location"
    weights
    "the locations that should be sampled first"
    start_locs
    "any samples taken previously (default empty)"
    prior_samples = Sample[]
    "whether noise should be learned or not (default false)"
    noise = false
    "the kernel to be used in the belief model (default multiKernel)"
    kernel = multiKernel
end

"""
The main function that runs the adaptive sampling routine. For each iteration, a
sample location is selected, a sample is collected, the belief model is updated,
and visuals are possibly shown. The run finishes when the designated number of
samples is collected.

Inputs:
- `samples`: a vector of samples, this can be used to jump-start a mission
  or resume a previous mission (default empty)
- `beliefs`: a vector of beliefs, this pairs with the previous argument
  (default empty)
- `visuals`: true or false to cause map plots to be shown or not (default false)
- `sleep_time`: the amount of time to wait after each iteration, useful for
  visualizations (default 0)

Outputs:
- `samples`: a vector of new samples collected
- `beliefs`: a vector of probabilistic representations of the quantities being
  searched for, one for each sample collection

# Examples
```julia
using Missions: simMission

mission = simMission(num_samples=10) # create the specific mission
samples, beliefs = mission(visuals=true, sleep_time=0.5) # run the mission
```
"""
function (M::Mission)(func=Returns(nothing);
                      samples=Sample[], beliefs=BeliefModel[],
                      seed_val=0, sleep_time=0)

    # initialize
    seed!(seed_val)
    new_loc = !isempty(M.start_locs) ? M.start_locs[1] : randomPoint(M.occupancy)

    lb, ub = bounds(M.occupancy)
    quantities = eachindex(M.sampler) # all current available quantities

    println("Mission started")

    for i in 1:M.num_samples
        println()
        println("Sample number $i")

        println("Next sample location: $new_loc")

        # prevent invalid locations
        M.occupancy(new_loc) && error("sample location is within obstacle")

        # sample all quantities
        new_samples = takeSamples(new_loc, M.sampler)
        append!(samples, new_samples)
        println("Sample values: $(getfield.(new_samples, :y))")

        # new belief
        beliefModel = BeliefModel([M.prior_samples; samples], lb, ub; M.noise, M.kernel)
        push!(beliefs, beliefModel)

        # new sample location
        sampleCost = nothing
        new_loc = nothing
        if i < M.num_samples
            if i < length(M.start_locs)
                new_loc = M.start_locs[i+1]
            else
                sampleCost = M.sampleCostType(M, samples, beliefModel, quantities)
                new_loc = selectSampleLocation(sampleCost, lb, ub)

                @debug "cost function values: $(Tuple(values(sampleCost, new_loc)))"
                @debug "cost function weights: $(Tuple(M.weights))"
                @debug "cost function terms: $(Tuple(values(sampleCost, new_loc)) .* Tuple(M.weights))"
                @debug "cost function value: $(sampleCost(new_loc))"
            end
        end

        # user-defined function (visualization, saving, etc.)
        func(M, samples, beliefModel, sampleCost, new_loc)
        @debug "output correlation matrix:" outputCorMat(beliefModel)
        sleep(sleep_time)
    end

    println()
    println("Mission complete")

    return samples, beliefs
end

function replay(M::Mission, full_samples, beliefs; func=Returns(nothing), sleep_time=0.0)
    quantities = eachindex(M.sampler) # all current available quantities

    println("Mission started")

    for i in eachindex(full_samples)
        println()
        println("Sample number $i")

        sample = full_samples[i]

        new_loc = sample.x[1]
        println("Next sample location: $new_loc")

        new_vals = sample.y
        println("Sample values: $(new_vals)")

        samples = full_samples[1:i]

        beliefModel = beliefs[i]
        sampleCost = M.sampleCostType(M, samples, beliefModel, quantities)

        new_loc = i < M.num_samples ? full_samples[i+1].x[1] : nothing
        func(M, samples, beliefModel, sampleCost, new_loc)
        @debug "output correlation matrix:" outputCorMat(beliefModel)
        sleep(sleep_time)
    end

    println()
    println("Mission complete")
end

function replay(M::Mission, full_samples; sleep_time=0.0)
    beliefs = map(1:length(full_samples)) do i
        BeliefModel([M.prior_samples; full_samples[1:i]], bounds(M.occupancy)...)
    end
    replay(M, full_samples, beliefs; sleep_time)
end

end
