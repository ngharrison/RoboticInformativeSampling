module InformativeSampling

# put all the code into the top namespace
include("Maps.jl")
include("Samples.jl")
include("SampleCosts.jl")
include("ROSInterface.jl")
include("Missions.jl")

export Maps, SampleCosts, Samples, ROSInterface, Missions

end
