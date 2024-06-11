
using InformativeSampling
using .Samples: takeSamples
using .ROSInterface: ROSConnection

# the topics that will be listened to for measurements
data_topics = [
    # Crop height avg in frame (excluding wheels)
    "/rss/gp/crop_height_avg",
    "/rss/silios/ndvi_bw_gradient_image" # TODO check this
]

done_topic = "sortie_finished"
pub_topic = "latest_sample"

sampler = ROSConnection(data_topics, done_topic, pub_topic)

# 50x50 meter space (alt2)
lower = [284725.0, 6241345.0]
upper = [284775.0, 6241395.0]
bounds = (; lower, upper)

n = (50, 50) # number of samples in each dimension
axs_sp = range.(bounds..., n)
locs = vec(collect.(Iterators.product(axs_sp...)))

samples = [takeSamples.(locs, Ref(sampler))...;]
