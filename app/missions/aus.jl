
using Logging: global_logger, ConsoleLogger, Info, Debug
using LinearAlgebra: norm
using DelimitedFiles: readdlm
using Statistics: cor
using Random: seed!

using InformativeSampling
using .Maps: Map, getBounds
using .Samples: Sample, MapsSampler, selectSampleLocation
using .SampleCosts: EIGFSampleCost
using .Missions: Mission

using InformativeSamplingUtils
using .Visualization: vis
using .DataIO: normalize, maps_dir, imgToMap

function ausMission(; seed_val=0, num_samples=30, priors=Bool[1,1,1])
    # have it run around australia

    seed!(seed_val)

    file_names = [
        "vege_ave_aus.csv",
        "topo_ave_aus.csv",
        "temp_ave_aus.csv",
        "rain_ave_aus.csv"
    ]

    images = readdlm.(maps_dir .* file_names, ',')

    bounds = (lower = [0.0, 0.0], upper = [1.0, 1.0])

    map0 = imgToMap(normalize(images[1]), bounds)
    sampler = MapsSampler(map0)

    prior_maps = [imgToMap(normalize(img), bounds) for img in images[2:end]]

    occupancy = imgToMap(Matrix{Bool}(reduce(.|, [isnan.(i)
                                                  for i in images])), bounds)

    sampleCostType = EIGFSampleCost

    ## initialize alg values
    # weights = [1e-1, 6, 5e-1, 3e-3] # mean, std, dist, prox
    # weights = (; μ=1, σ=5e3, τ=1, d=1) # others
    weights = (; μ=1, σ=1e2, τ=1, d=1) # others
    start_locs = [[0.8, 0.6]] # starting locations


    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations

    # maximize minimum distance between samples
    points_sp = Vector{Float64}[]
    sampleCost = x -> occupancy(x) ? Inf : -minimum(norm(loc - x) for loc in points_sp; init=Inf)
    for _ in 1:25
        x = selectSampleLocation(sampleCost, getBounds(occupancy)...)
        push!(points_sp, x)
        # x, v = rand(occupancy)
        # !v && push!(points_sp, x)
    end
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
mission, prior_maps = ausMission(num_samples=10)

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
    mission, _ = ausMission(priors=collect(Bool, priors))
    # empty!(mission.prior_samples)

    ## run search alg
    @time samples, beliefs = mission(vis, sleep_time=0.0);
    @debug "output correlation matrix:" outputCorMat(beliefs[end])
    # save(mission, samples, beliefs; animation=true)

    ## calculate errors
    metrics = calcMetrics(mission, beliefs, 1)

    ## save outputs
    save(metrics; file_name="aus_ave_means_noise/metrics_$(join(priors))")
    save(mission, samples, beliefs; file_name="aus_ave_means_noise/mission_$(join(priors))")
end
