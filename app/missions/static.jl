
using GridMaps: GridMap, generateAxes

using InformativeSampling
using .Samples: Sample, takeSamples

include(dirname(Base.active_project()) * "/ros/ROSInterface.jl")
using .ROSInterface: ROSSampler

# the topics that will be listened to for measurements
data_topics = [
    "value1", "value2"
]

done_topic = "sortie_finished"
pub_topic = "latest_sample"

sampler = ROSSampler(data_topics, done_topic, pub_topic)

bounds = (lower = [0.0, 0.0], upper = [1.0, 1.0])

n = (3, 3) # number of samples in each dimension
axs_sp = range.(bounds..., n)
locs = vec(collect.(Iterators.product(axs_sp...)))

# start other scripts
cmds = (`$(Base.source_dir())/../ros/sample_sim.py`,
        `$(Base.source_dir())/../ros/sortie_sim.py`)

procs = run.(cmds; wait=false)

try
    samples = [takeSamples.(locs, Ref(sampler))...;]
    @show samples
finally
    # kill other scripts
    kill.(procs)
end
