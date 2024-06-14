
using Logging: global_logger, ConsoleLogger, Info, Debug

using InformativeSampling
using .Samples: takeSamples
using .ROSInterface: ROSConnection

using InformativeSamplingUtils
using .DataIO: save

# the topics that will be listened to for measurements
data_topics = [
    # Crop height avg in frame (excluding wheels)
    "/rss/gp/crop_height_avg"
    "/rss/silios/ndvi_avg"
]

done_topic = "sortie_finished"
pub_topic = "latest_sample"

sampler = ROSConnection(data_topics, done_topic, pub_topic)

# 50x50 meter space (alt2)
lower = [284725.0, 6241345.0]
upper = [284775.0, 6241395.0]
bounds = (; lower, upper)

# grid
n = (25, 25) # number of samples in each dimension
axs_sp = range.(bounds..., n)
temp = vec(collect.(Iterators.product(axs_sp...)))

# flip every other
arrs = Iterators.partition(temp, n[1])
locs = collect(Iterators.flatmap(enumerate(arrs)) do (i, arr)
    (i % 2 == 1 ? identity : reverse)(arr)
end)

global_logger(ConsoleLogger(stderr, Debug))

samples = []
for loc in locs[299:end]
    samples_part = takeSamples(loc, sampler)
    save(samples_part; sub_dir_name="pye_farm_trial2/dense")
    append!(samples, samples_part)
end

save(samples; sub_dir_name="pye_farm_trial2/dense")
