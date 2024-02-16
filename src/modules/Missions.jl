module Missions

using LinearAlgebra: I, norm
using Images: load, imresize, Gray, gray
using DelimitedFiles: readdlm
using Statistics: mean, cor
using Random: seed!
using DocStringExtensions: TYPEDSIGNATURES, TYPEDFIELDS

using Maps: Map, imgToMap, GaussGroundTruth, Peak, pointToCell, cellToPoint, generateAxes
using Samples: Sample, MapsSampler, selectSampleLocation, takeSamples
using SampleCosts: SampleCost, values, BasicSampleCost,
                   NormedSampleCost, MIPTSampleCost, EIGFSampleCost
using BeliefModels: BeliefModel, outputCorMat
using Visualization: visualize

export simMission, ausMission, nswMission, conradMission, rosMission,
       pyeFarmMission, Mission, maps_dir

const maps_dir = dirname(Base.active_project()) * "/maps/"

"""
Fields:
$(TYPEDFIELDS)

Defined as a keyword struct, so all arguments are passed in as keywords:
```julia
mission = Mission(; occupancy,
                  sampler,
                  num_samples,
                  sampleCostType,
                  weights,
                  start_loc,
                  prior_samples)
```
"""
@kwdef struct Mission
    "an occupancy map, true in cells that are occupied"
    occupancy
    "a function that returns a measurement value for any input"
    sampler
    "the number of samples to collect in one run"
    num_samples
    "a constructor for the function that returns the (negated) value of taking a sample"
    sampleCostType
    "weights for picking the next sample location"
    weights
    "the starting location"
    start_loc
    "any samples taken previously (default empty)"
    prior_samples = Sample[]
end

"""
The main function that runs the adaptive sampling routine. For each iteration, a
sample location is selected, a sample is collected, the belief model is updated,
and visuals are possibly shown. The run finishes when the designated number of
samples is collected.

Inputs:
- `samples`: a vector of samples, this can be used to jump-start a mission
  or resume a previous mission (default empty)
- `beliefs`: a vector of beliefs, this pairs with the previous argument
  (default empty)
- `visuals`: true or false to cause map plots to be shown or not (default false)
- `sleep_time`: the amount of time to wait after each iteration, useful for
  visualizations (default 0)

Outputs:
- `samples`: a vector of new samples collected
- `beliefs`: a vector of probabilistic representations of the quantities being
  searched for, one for each sample collection

# Examples
```julia
using Missions: simMission

mission = simMission(num_samples=10) # create the specific mission
samples, beliefs = mission(visuals=true, sleep_time=0.5) # run the mission
```
"""
function (M::Mission)(func=Returns(nothing);
                      samples=Sample[], beliefs=BeliefModel[],
                      seed_val=0, sleep_time=0)
    M.occupancy(M.start_loc) && error("start location is within obstacle")

    # initialize
    seed!(seed_val)
    lb, ub = M.occupancy.lb, M.occupancy.ub
    new_loc = M.start_loc
    quantities = eachindex(M.sampler) # all current available quantities

    beliefModel = nothing
    sampleCost = nothing

    println("Mission started")

    for i in 1:M.num_samples
        if !isempty(samples)
            # new belief
            beliefModel = BeliefModel([M.prior_samples; samples], lb, ub)
            push!(beliefs, beliefModel)

            # new sample location
            sampleCost = M.sampleCostType(M, samples, beliefModel, quantities)
            new_loc = selectSampleLocation(sampleCost, lb, ub)

            @debug "cost function values: $(Tuple(values(sampleCost, new_loc)))"
            @debug "cost function weights: $(Tuple(M.weights))"
            @debug "cost function terms: $(Tuple(values(sampleCost, new_loc)) .* Tuple(M.weights))"
            @debug "cost function value: $(sampleCost(new_loc))"

            # user-defined function (visualization, saving, etc.)
            func(M, samples, beliefModel, sampleCost, new_loc)
            @debug "output determination matrix:" outputCorMat(beliefs[end]).^2
            sleep(sleep_time)
        end

        println()
        println("Sample number $i")

        println("Next sample location: $new_loc")

        # sample all quantities
        new_samples = takeSamples(new_loc, M.sampler)
        append!(samples, new_samples)
        println("Sample values: $(getfield.(new_samples, :y))")
    end

    # new belief
    beliefModel = BeliefModel([M.prior_samples; samples], lb, ub)
    push!(beliefs, beliefModel)

    # user-defined function (visualization, saving, etc.)
    func(M, samples, beliefModel, sampleCost, new_loc)
    @debug "output determination matrix:" outputCorMat(beliefs[end]).^2

    println()
    println("Mission complete")

    return samples, beliefs
end

# Constructors for Mission data
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
    weights = (; μ=1, σ=5e2, τ=1, d=0) # others
    # weights = (; μ=1, σ=1, τ=.1, d=1)
    start_loc = [1.0, 0.0] # starting location


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

    display(visualize(sampler.maps..., prior_maps...;
                      samples=points_sp,
                      titles=["QOI", "Scaling Factor", "Additive Noise", "Random Map"]))

    return Mission(; occupancy,
                   sampler,
                   num_samples,
                   sampleCostType,
                   weights,
                   start_loc,
                   prior_samples)#, [cor(vec(map0), vec(d)).^2 for d in prior_maps]

end

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
    weights = (; μ=1, σ=5e3, τ=1, d=1) # others
    start_loc = [0.8, 0.6] # starting location


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

    display(visualize(sampler.maps..., prior_maps...;
              titles=["Vegetation", "Elevation", "Ground Temperature", "Rainfall"],
              samples=points_sp))

    return Mission(; occupancy,
                   sampler,
                   num_samples,
                   sampleCostType,
                   weights,
                   start_loc,
                   prior_samples)
end

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

    ims_sm = spatialAve.(images, 3)

    lb = [0.0, 0.0]; ub = [1.0, 1.0]

    map0 = imgToMap(normalize(ims_sm[1]), lb, ub)
    sampler = MapsSampler(map0)

    prior_maps = [imgToMap(normalize(img), lb, ub) for img in ims_sm[2:end]]

    occupancy = imgToMap(Matrix{Bool}(reduce(.|, [isnan.(i)
                                                  for i in ims_sm])), lb, ub)

    sampleCostType = EIGFSampleCost

    ## initialize alg values
    # weights = [1e-1, 6, 5e-1, 3e-3] # mean, std, dist, prox
    # weights = (; μ=1, σ=5e3, τ=1, d=1) # others
    weights = (; μ=1, σ=5e3, τ=1, d=1) # others
    start_loc = [1.0, 0.0] # starting location


    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations

    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations
    n = (5,5) # number of samples in each dimension
    axs_sp = range.(lb, ub, n)
    points_sp = vec(collect.(Iterators.product(axs_sp...)))
    prior_samples = [Sample((x, i+length(sampler)), d(x))
                     for (i, d) in enumerate(prior_maps[priors])
                         for x in points_sp if !isnan(d(x))]

    # Calculate correlation coefficients
    [cor(map0.(points_sp), d.(points_sp)) for d in prior_maps]
    [cor(vec(map0[.!occupancy]), vec(d[.!occupancy])) for d in prior_maps]
    # scatter(vec(map0[.!occupancy]), [vec(d[.!occupancy]) for d in prior_maps], layout=3)

    display(visualize(sampler.maps..., prior_maps...;
              titles=["Vegetation", "Elevation", "Ground Temperature", "Rainfall"],
              samples=points_sp))

    return Mission(; occupancy,
                   sampler,
                   num_samples,
                   sampleCostType,
                   weights,
                   start_loc,
                   prior_samples)


end

function conradMission()

    file_names = [
        maps_dir * "weightMap1.csv",
        maps_dir * "weightMap2.csv",
    ]

    # need to drop the last point
    data = readdlm.(maps_dir .* file_names, ',')

    lb = minimum(minimum(d, dims=1)[1:2] for d in data)
    ub = maximum(maximum(d, dims=1)[1:2] for d in data)

    # scatter(data[1][:,1], data[1][:,2], marker_z=data[1][:,3])
    # scatter(data[2][:,1], data[2][:,2], marker_z=data[2][:,3])

    n = floor(Int, sqrt(maximum(size.(data, 1))))

    maps = [Map(zeros(n, n), lb, ub) for _ in 1:2]

    for (x,y,z) in eachrow(data[1])
        maps[1][pointToCell([x,y], maps[1])] = z
    end
    for (x,y,z) in eachrow(data[2])
        maps[2][pointToCell([x,y], maps[2])] = z
    end

    sampler = MapsSampler(maps)

    sampleCostType = NormedSampleCost

    ## initialize alg values
    weights = [1, 6, 1, 3e-3] # mean, std, dist, prox
    start_loc = [0.8, 0.6] # starting location
    num_samples = 20

    occupancy = Map(zeros(Bool, n, n), lb, ub)

    return Mission(; occupancy,
                   sampler,
                   num_samples,
                   sampleCostType,
                   weights,
                   start_loc)
end

function rosMission(; num_samples=4)

    # this allows loading a module from within this function
    # it runs this block in the global namespace of this module
    # the benefit is that it only gets used if this function is run
    @eval begin
        # this requires a working rospy installation
        using ROSInterface: ROSConnection

        # the topics that will be listened to for measurements
        sub_topics = [
            # Crop height avg and std in frame (excluding wheels)
            ("value1", "value2")
        ]

        sampler = ROSConnection(sub_topics)
    end

    lb = [1.0, 1.0]; ub = [9.0, 9.0]

    occupancy = Map(zeros(Bool, 100, 100), lb, ub)

    sampleCostType = EIGFSampleCost

    ## initialize alg values
    weights = (; μ=1, σ=5e3, τ=1, d=1) # mean, std, dist, prox
    start_loc = [1.0, 1.0]

    return Mission(; occupancy,
                   sampler,
                   num_samples,
                   sampleCostType,
                   weights,
                   start_loc)
end

function pyeFarmMission(; num_samples=4)

    # this allows loading a module from within this function
    # it runs this block in the global namespace of this module
    # the benefit is that it only gets used if this function is run
    @eval begin
        # this requires a working rospy installation
        using ROSInterface: ROSConnection

        # the topics that will be listened to for measurements
        sub_topics = [
            # Crop height avg and std in frame (excluding wheels)
            ("/rss/gp/crop_height_avg", "/rss/gp/crop_height_std")
        ]

        sampler = ROSConnection(sub_topics)
    end

    # # full bounds
    # lb = [284711.12, 6241319.42]
    # ub = [284802.91, 6241403.93]
    # lb = [284698., 6241343.]
    # ub = [284805., 6241405.]

    # smaller bounds
    # lb = [284729., 6241334.]
    # ub = [284742., 6241348.]
    # lb = [284724., 6241344.]
    # ub = lb .+ 10

    # full space (alt2)
    lb = [284725., 6241345.]
    ub = [284775., 6241395.]

    elev_img = load(maps_dir * "iros_alt2_dem.tif")
    elevMap = imgToMap(gray.(elev_img), lb, ub)

    # # small patch (alt3)
    # lb = [284745., 6241345.]
    # ub = [284760., 6241360.]
    #
    # elev_img = load(maps_dir * "iros_alt3_dem.tif")
    # elevMap = imgToMap(gray.(elev_img), lb, ub)

    prior_maps = [elevMap]

    occupancy = Map(zeros(Bool, 100, 100), lb, ub)

    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations
    n = (7,7) # number of samples in each dimension
    axs_sp = range.(lb, ub, n)
    points_sp = vec(collect.(Iterators.product(axs_sp...)))
    prior_samples = [Sample{Float64}((x, i+length(sampler)), d(x))
                     for (i, d) in enumerate(prior_maps)
                         for x in points_sp if !isnan(d(x))]
    prior_samples = Sample{Float64}[]

    display(visualize(elevMap; samples=points_sp))

    num_samples = 30

    sampleCostType = EIGFSampleCost

    ## initialize alg values
    weights = (; μ=1, σ=5e1, τ=1, d=1) # mean, std, dist, prox
    start_loc = lb .+ 2

    return Mission(; occupancy,
                   sampler,
                   num_samples,
                   sampleCostType,
                   weights,
                   start_loc,
                   prior_samples)
end

# helper methods
function normalize(a)
    l, h = extrema(filter(!isnan, a))
    return (a .- l) ./ (h - l)
end

function spatialAve(M, extent=1)
    N = zero(M)
    for i in axes(M,1), j in axes(M,2)
        tot = 0
        count = 0
        for k in -extent:extent, l in -extent:extent
            m = i + k
            n = j + l
            if 1 <= m <= size(M,1) && 1 <= n <= size(M,2) && !isnan(M[m,n])
                tot += M[m,n]
                count += 1
            end
        end
        N[i,j] = tot/count
    end
    return N
end

end
