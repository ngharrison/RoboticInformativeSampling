module InformativeSampling

# put all the code into the top namespace
include("Samples.jl")
include("SampleCosts.jl")
include("ROSInterface.jl")
include("Missions.jl")

export SampleCosts, Samples, ROSInterface, Missions

end
