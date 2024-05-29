#!/usr/bin/env julia

using Pkg
Pkg.activate(Base.source_dir() * "/..")

using RobotOS
@rosimport informative_sampling.srv: GenerateBeliefModel
@rosimport informative_sampling.msg: BeliefModelParameters
rostypegen()
using .informative_sampling.srv
using .informative_sampling.msg: BeliefModelParameters

using InformativeSampling
using .Samples, .BeliefModels

# callback function
function handleGenerateBeliefModel(req)
    println("Generating belief model")
    samples = map(req.samples) do s
        Sample((s.location, s.quantity_index), s.measurement)
    end
    bounds = (; req.bounds.lower, req.bounds.upper)
    bm = BeliefModel(samples, bounds)
    res = BeliefModelParameters(bm.θ.σ, bm.θ.ℓ, bm.θ.σn)
    return GenerateBeliefModelResponse(res)
end

function generateBeliefModelServer()
    init_node("generate_belief_model_server")
    s = Service("generate_belief_model", GenerateBeliefModel, handleGenerateBeliefModel)
    println("Ready to add generate belief model")
    spin()
end

generateBeliefModelServer()
