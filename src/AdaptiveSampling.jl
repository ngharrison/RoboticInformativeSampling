module AdaptiveSampling

# put all the code into the top namespace
include("Maps.jl")
include("Paths.jl")
include("SampleCosts.jl")
include("Samples.jl")
include("Kernels.jl")
include("BeliefModels.jl")
include("ROSInterface.jl")
include("Missions.jl")

include("Metrics.jl")
include("Visualization.jl")
include("Outputs.jl")

export Maps, Paths, SampleCosts, Samples, BeliefModels, Kernels,
       Metrics, Visualization, Outputs, ROSInterface, Missions

end
