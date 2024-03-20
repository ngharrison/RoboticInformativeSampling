
using LinearAlgebra: norm
using DelimitedFiles: readdlm
using Statistics: cor
using Random: seed!

using AdaptiveSampling: Maps, Samples, SampleCosts, Missions, Visualization

using .Maps: Map, imgToMap, maps_dir
using .Samples: Sample, MapsSampler, selectSampleLocation
using .SampleCosts: EIGFSampleCost
using .Missions: Mission
using .Visualization: vis

include("../utils/utils.jl")

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

    lb = [0.0, 0.0]; ub = [1.0, 1.0]

    map0 = imgToMap(normalize(images[1]), lb, ub)
    sampler = MapsSampler(map0)

    prior_maps = [imgToMap(normalize(img), lb, ub) for img in images[2:end]]

    occupancy = imgToMap(Matrix{Bool}(reduce(.|, [isnan.(i)
                                                  for i in images])), lb, ub)

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
        x = selectSampleLocation(sampleCost, occupancy.lb, occupancy.ub)
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

    vis(sampler.maps..., prior_maps...;
              titles=["Vegetation", "Elevation", "Ground Temperature", "Rainfall"],
              points=points_sp)

    return Mission(; occupancy,
                   sampler,
                   num_samples,
                   sampleCostType,
                   weights,
                   start_locs,
                   prior_samples)
end


#* Run

# set the logging level: Info or Debug
using Logging
global_logger(ConsoleLogger(stderr, Logging.Info))

using AdaptiveSampling: Visualization

using .Visualization: vis

## initialize data for mission
mission = ausMission(num_samples=10)

## run search alg
@time samples, beliefs = mission(
    vis;
    sleep_time=0.0
);


#* Pair

# set the logging level: Info or Debug
using Logging
global_logger(ConsoleLogger(stderr, Logging.Info))

using AdaptiveSampling: BeliefModels, Visualization, Metrics, Outputs

using .BeliefModels: outputCorMat
using .Visualization: vis
using .Metrics: calcMetrics
using .Outputs: save

@time for priors in [(0,0,0), (1,1,1)]
    ## initialize data for mission
    priors = (0,0,0)
    mission = ausMission(priors=collect(Bool, priors))
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
