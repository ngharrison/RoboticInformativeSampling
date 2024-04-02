
using Logging: global_logger, ConsoleLogger, Info, Debug
using LinearAlgebra: I, norm
using Statistics: mean, cor
using Random: seed!

using AdaptiveSampling

using .Maps: Map, GaussGroundTruth, Peak, generateAxes
using .Samples: Sample, MapsSampler
using .SampleCosts: EIGFSampleCost
using .Missions: Mission
using .Visualization: vis

function simMission(; seed_val=0, num_samples=30, num_peaks=3, priors=Bool[1,1,1])
    seed!(seed_val) # make random values deterministic

    lb = [0.0, 0.0]; ub = [1.0, 1.0]

    # # read in elevation
    # elev_img = load(maps_dir * "arthursleigh_shed_small.tif")
    # elevMap = imgToMap(gray.(elev_img), lb, ub)

    # # read in obstacles
    # obs_img = load(maps_dir * "obstacles_fieldsouth_220727.tif")
    # obs_img_res = imresize(obs_img, size(elev_img))
    # # the image we have has zeros for obstacles, need to flip
    # occ_mat = Matrix{Bool}(Gray.(obs_img_res) .== 0)
    # occupancy = imgToMap(occ_mat, lb, ub)

    occupancy = Map(zeros(Bool, 100, 100), lb, ub)

    ## initialize ground truth

    # simulated
    peaks = map(1:num_peaks) do _
        μ = rand(2).*(ub-lb) .+ lb
        Σ = 0.02*(rand()+0.5)*mean(ub-lb)^2*I
        h = rand()
        Peak(μ, Σ, h)
    end
    ggt = GaussGroundTruth(peaks)
    axs, points = generateAxes(occupancy)
    mat = ggt(points)
    map0 = Map(mat./maximum(mat), lb, ub)

    ## Create prior prior_samples

    # none -- leave uncommented
    prior_maps = []

    # multiplicative
    m = Map(abs.(map0 .* randn()), lb, ub)
    push!(prior_maps, m)

    # additive
    m = Map(abs.(map0 .+ 0.2 .* randn(size(map0))), lb, ub)
    push!(prior_maps, m)

    # # both
    # push!(prior_maps, Map(abs.(map0 .* randn() + 0.1 .* randn(size(map0))), lb, ub))

    # # spatial shift
    # t = rand(1:7)
    # push!(prior_maps, [zeros(size(map0,1),t) map0[:,1:end-t]]) # shift

    # random peaks
    peaks = map(1:num_peaks) do _
        μ = rand(2).*(ub-lb) .+ lb
        Σ = 0.02*(rand()+0.5)*mean(ub-lb)^2*I
        h = rand()
        Peak(μ, Σ, h)
    end
    tggt = GaussGroundTruth(peaks)
    tmat = tggt(points)
    m = Map(tmat./maximum(tmat), lb, ub)
    push!(prior_maps, m)

    # # purely random values
    # m = Map(rand(size(map0)...), lb, ub)
    # push!(prior_maps, m)

    sampler = MapsSampler(map0)

    sampleCostType = EIGFSampleCost

    ## initialize alg values
    # weights = (; μ=17, σ=1.5, τ=7)
    # weights = (; μ=3, σ=1, τ=.5, d=1)
    # weights = (; μ=1, σ=1e1, τ=1, d=0) # sogp
    weights = (; μ=1, σ=1e2, τ=1, d=0) # others
    # weights = (; μ=1, σ=1, τ=.1, d=1)
    start_locs = [[1.0, 0.0]] # starting location

    # n = (4,4) # number of samples in each dimension
    # axs_sp = range.(lb, ub, n)
    # start_locs = vec(collect.(Iterators.product(axs_sp...))) # starting locations


    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations
    n = (5,5) # number of samples in each dimension
    axs_sp = range.(lb, ub, n)
    points_sp = vec(collect.(Iterators.product(axs_sp...)))
    prior_samples = [Sample((x, i+length(sampler)), d(x))
                     for (i, d) in enumerate(prior_maps[priors])
                         for x in points_sp if !isnan(d(x))]

    # Calculate correlation coefficients
    @debug [cor(map0.(points_sp), d.(points_sp)) for d in prior_maps]
    # @debug [cor(vec(map0), vec(d)) for d in prior_maps]

    # vis(sampler.maps..., prior_maps...;
    #                   points=points_sp,
    #                   titles=["QOI", "Scaling Factor", "Additive Noise", "Random Map"])

    noise = (0.0, :learned)

    return Mission(; occupancy,
                   sampler,
                   num_samples,
                   sampleCostType,
                   weights,
                   start_locs,
                   prior_samples,
                   noise)

end


#* Run

# set the logging level: Info or Debug
global_logger(ConsoleLogger(stderr, Info))

using AdaptiveSampling

using .Visualization: vis

## initialize data for mission
mission = simMission(num_samples=10)

## run search alg
@time samples, beliefs = mission(
    vis;
    sleep_time=0.0
);


# using Metrics: calcMetrics
# using Outputs: save
#
# ## calculate errors
# metrics = calcMetrics(mission, beliefs, 1)
#
# ## save outputs
# save(mission, samples, beliefs; animation=true)
# save(metrics)


#* Batch

# set the logging level: Info or Debug
global_logger(ConsoleLogger(stderr, Info))

using .BeliefModels: outputCorMat
using .Visualization: visualize
using .Metrics: calcMetrics
using .Outputs: save

mission_peaks = [3,3,4,4,5,5]
num_runs = 3
metrics = Array{Any, 2}(undef, (length(mission_peaks), num_runs))
# pick all the prior data combinations
@time for priors in Iterators.product(fill(0:1,3)...)
    for (i, num_peaks) in enumerate(mission_peaks)
        ## initialize data for mission
        mission = simMission(; seed_val=i, num_peaks, priors=collect(Bool, priors))
        for j in 1:num_runs
            ## run search alg
            @time samples, beliefs = mission(seed_val=j, sleep_time=0.0);
            @debug "output correlation matrix:" outputCorMat(beliefs[end])
            # save(mission, samples, beliefs; animation=true)
            ## calculate errors
            metrics[i,j] = calcMetrics(mission, beliefs, 1)
        end
    end
    ## save outputs
    save(metrics; file_name="batch_means_noise_1e2/metrics_$(join(priors))")
end
