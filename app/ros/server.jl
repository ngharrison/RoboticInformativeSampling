#!/usr/bin/env julia

using Pkg
Pkg.activate(Base.source_dir() * "/..")

using RobotOS
@rosimport informative_sampling.srv: GenerateBeliefModel
@rosimport informative_sampling.srv: GenerateBeliefMaps
@rosimport informative_sampling.msg: BeliefModelParameters
@rosimport std_msgs.msg: Float64MultiArray, MultiArrayLayout, MultiArrayDimension
rostypegen()
using .informative_sampling.srv
using .informative_sampling.msg: BeliefModelParameters
using .std_msgs.msg: Float64MultiArray, MultiArrayLayout, MultiArrayDimension

using InformativeSampling
using .Samples, .BeliefModels, .Maps

# takes in a function and returns a new function that does the same thing, but
# wraps it in a try-catch block, so errors are caught and printed
function catchErrors(func)
    return req -> begin
        try
            func(req)
        catch err
            println(err)
        end
    end
end

# callback functions
function handleGenerateBeliefModel(req)
    println("Generating belief model")

    samples = map(req.samples) do s
        Sample((s.location, s.quantity_index), s.measurement)
    end
    bounds = (; req.bounds.lower, req.bounds.upper)
    bm = BeliefModel(samples, bounds)

    params = BeliefModelParameters(bm.θ.σ, bm.θ.ℓ, bm.θ.σn)
    return GenerateBeliefModelResponse(params)
end

function handleGenerateBeliefMaps(req)
    println("Generating belief maps")

    samples = map(req.samples) do s
        Sample((s.location, s.quantity_index), s.measurement)
    end
    bounds = (; req.bounds.lower, req.bounds.upper)

    beliefModel = BeliefModel(samples, bounds)

    dims = req.dims
    quantity = req.quantity_index

    _, points = generateAxes(bounds, Tuple(dims))
    μ, σ = beliefModel(tuple.(vec(points), quantity))

    belief_map, uncertainty_map = (
        Float64MultiArray(
            MultiArrayLayout(MultiArrayDimension.("", dims, 1), 0),
            data
        )
        for data in (μ, σ)
    )

    return GenerateBeliefMapsResponse(belief_map, uncertainty_map)
end

function main()
    init_node("generate_belief_model_server")

    # create services
    Service("generate_belief_model", GenerateBeliefModel,
            catchErrors(handleGenerateBeliefModel))
    Service("generate_belief_maps", GenerateBeliefMaps,
            catchErrors(handleGenerateBeliefMaps))

    # wait for requests
    println("Ready to serve")
    spin()
end

main()
