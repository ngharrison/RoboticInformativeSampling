
using Logging: global_logger, ConsoleLogger, Info, Debug
using DelimitedFiles: readdlm
using Statistics: cor
using Random: seed!

using InformativeSampling
using .Maps: Map
using .Samples: Sample, MapsSampler
using .SampleCosts: EIGF
using .Missions: Mission

using InformativeSamplingUtils
using .Visualization: vis
using .DataIO: normalize, maps_dir, imgToMap

function nswMission(; seed_val=0, num_samples=30, priors=Bool[1,1,1])
    # have it run around australia

    seed!(seed_val)

    file_names = [
        "vege_ave_nsw.csv",
        "topo_ave_nsw.csv",
        "temp_ave_nsw.csv",
        "rain_ave_nsw.csv"
    ]

    images = readdlm.(maps_dir .* file_names, ',')

    ims_sm = images

    bounds = (lower = [0.0, 0.0], upper = [1.0, 1.0])

    map0 = imgToMap(normalize(ims_sm[1]), bounds)
    sampler = MapsSampler(map0)

    prior_maps = [imgToMap(normalize(img), bounds) for img in ims_sm[2:end]]

    occupancy = imgToMap(Matrix{Bool}(reduce(.|, [isnan.(i)
                                                  for i in ims_sm])), bounds)

    sampleCostType = EIGF

    ## initialize alg values
    # weights = [1e-1, 6, 5e-1, 3e-3] # mean, std, dist, prox
    # weights = (; μ=1, σ=5e3, τ=1, d=1) # others
    weights = (; μ=1, σ=5e2, τ=1, d=1) # others
    start_locs = [[1.0, 0.0]] # starting locations


    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations

    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations
    n = (5,5) # number of samples in each dimension
    axs_sp = range.(bounds..., n)
    points_sp = vec(collect.(Iterators.product(axs_sp...)))
    prior_samples = [Sample((x, i+length(sampler)), d(x))
                     for (i, d) in enumerate(prior_maps[priors])
                         for x in points_sp if !isnan(d(x))]

    # Calculate correlation coefficients
    [cor(map0.(points_sp), d.(points_sp)) for d in prior_maps]
    [cor(vec(map0[.!occupancy]), vec(d[.!occupancy])) for d in prior_maps]
    # scatter(vec(map0[.!occupancy]), [vec(d[.!occupancy]) for d in prior_maps], layout=3)

    mission = Mission(;
        occupancy,
        sampler,
        num_samples,
        sampleCostType,
        weights,
        start_locs,
        prior_samples
    )

    return mission, prior_maps


end


#* Run

# set the logging level: Info or Debug
global_logger(ConsoleLogger(stderr, Info))

## initialize data for mission
mission, prior_maps = nswMission(num_samples=10)

vis(mission.sampler..., prior_maps...;
    titles=["Vegetation", "Elevation", "Ground Temperature", "Rainfall"],
    points=first.(getfield.(mission.prior_samples, :x)))

## run search alg
@time samples, beliefs = mission(
    vis;
    sleep_time=0.0
);


#* Pair

# set the logging level: Info or Debug
global_logger(ConsoleLogger(stderr, Info))

using .BeliefModels: outputCorMat

using .Metrics: calcMetrics
using .DataIO: save

@time for priors in [(0,0,0), (1,1,1)]
    ## initialize data for mission
    priors = (0,0,0)
    mission, _ = nswMission(priors=collect(Bool, priors))
    # empty!(mission.prior_samples)

    ## run search alg
    @time samples, beliefs = mission(vis, sleep_time=0.0);
    @debug "output correlation matrix:" outputCorMat(beliefs[end])
    # save(mission, samples, beliefs; animation=true)

    ## calculate errors
    metrics = calcMetrics(mission, samples, beliefs, 1)

    ## save outputs
    save(metrics; file_name="aus_patch_ave_means_noise/metrics_$(join(priors))")
    save(mission, samples, beliefs; file_name="aus_patch_ave_means_noise/mission_$(join(priors))")
end
