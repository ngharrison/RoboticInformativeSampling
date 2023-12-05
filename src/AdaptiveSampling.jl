module AdaptiveSampling

# put all the code into the top namespace
include("modules/Maps.jl")
include("modules/Paths.jl")
include("modules/SampleCosts.jl")
include("modules/Samples.jl")
include("modules/BeliefModels.jl")
include("modules/ROSInterface.jl")
include("modules/Missions.jl")

include("modules/Metrics.jl")
include("modules/Visualization.jl")
include("modules/Outputs.jl")

export Maps, Paths, SampleCosts, Samples, BeliefModels,
       Metrics, Visualization, Outputs, ROSInterface, Missions

end
