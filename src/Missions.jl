module Missions

using LinearAlgebra: I
using Images: load, imresize, Gray, gray
using DelimitedFiles: readdlm
using Statistics: cor
using Random: seed!
using DocStringExtensions: SIGNATURES

using Maps: Map, imgToMap, GaussGroundTruth, MultiMap, Peak, pointToCell
using Samples: Sample, selectSampleLocation, takeSamples
using SampleCosts: SampleCost, values, BasicSampleCost,
                   NormedSampleCost, MIPTSampleCost, EIGFSampleCost
using BeliefModels: BeliefModel, outputCorMat
using Visualization: visualize

const maps_dir = dirname(Base.active_project()) * "/maps/"

"""
Inputs:
    - occupancy: an occupancy map, true in cells that are occupied
    - sampler: a function that returns a measurement value at any point
    - start_loc: the starting location
    - weights: weights for picking the next sample location
    - num_samples: the number of samples to collect in one run (default 20)
    - prior_samples: any samples taken previously (default empty)
    - samples: any samples taken already during the current mission (default empty)
"""
@kwdef struct Mission
    occupancy
    sampler
    start_loc
    weights
    num_samples
    prior_samples = Sample[]
end

# Constructors for Mission data
function simMission()
    seed!(2) # make random values deterministic

    lb = [0.0, 0.0]; ub = [1.0, 1.0]

    # read in elevation
    elev_img = load(maps_dir * "arthursleigh_shed_small.tif")
    elevMap = imgToMap(gray.(elev_img), lb, ub)

    # read in obstacles
    obs_img = load(maps_dir * "obstacles_fieldsouth_220727.tif")
    obs_img_res = imresize(obs_img, size(elev_img))
    # the image we have has zeros for obstacles, need to flip
    occ_mat = Matrix{Bool}(Gray.(obs_img_res) .== 0')
    occupancy = imgToMap(occ_mat, lb, ub)

    ## initialize ground truth

    # simulated
    peaks = [Peak([0.3, 0.3], 0.03*I, 1.0),
             Peak([0.8, 0.7], 0.008*I, 0.4)]
    ggt = GaussGroundTruth(peaks)
    axs = range.(lb, ub, size(elev_img))
    points = collect.(Iterators.product(axs...))
    map0 = Map(ggt(points), lb, ub)

    ## Create prior prior_samples

    # none -- leave uncommented
    prior_maps = []

    # additive
    push!(prior_maps, Map(abs.(map0 .+ 0.1 .* randn(size(map0))), lb, ub))

    # multiplicative
    push!(prior_maps, Map(abs.(map0 .* randn()), lb, ub))

    # # both
    # push!(prior_maps, Map(abs.(map0 .* randn() + 0.1 .* randn(size(map0))), lb, ub))

    # # spatial shift
    # t = rand(1:7)
    # push!(prior_maps, [zeros(size(map0,1),t) map0[:,1:end-t]]) # shift

    # purely random
    num_peaks = 3
    peaks = [Peak(rand(2).*(ub-lb) .+ lb, 0.02*I, rand())
             for i in 1:num_peaks]
    tggt = GaussGroundTruth(peaks)
    # push!(prior_maps, Map(tggt(points), lb, ub))

    sampler = MultiMap(map0, Map(tggt(points), lb, ub))

    ## initialize alg values
    # weights = (; μ=17, σ=1.5, τ=7)
    weights = (; μ=3, σ=1, τ=.5, d=1)
    # weights = (; μ=1, σ=1, τ=1, d=1)
    # weights = (; μ=1, σ=1, τ=.1, d=1)
    start_loc = [0.5, 0.2] # starting location
    num_samples = 20


    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations
    n = (5,5) # number of samples in each dimension
    axs_sp = range.(lb, ub, n)
    points_sp = vec(collect.(Iterators.product(axs_sp...)))
    prior_samples = [Sample((x, i+length(sampler)), d(x))
                     for (i, d) in enumerate(prior_maps)
                         for x in points_sp if !isnan(d(x))]

    # Calculate correlation coefficients
    [cor(map0.(points_sp), d.(points_sp)) for d in prior_maps]

    visualize(sampler.maps..., prior_maps...;
              titles=["Ground Truth 1", "Ground Truth 2", "Prior 1", "Prior 2"],
              samples=points_sp)

    return Mission(; occupancy,
                       sampler,
                       start_loc,
                       weights,
                       num_samples,
                       prior_samples)

end


normalize(a) = a ./ maximum(filter(!isnan, a))


function ausMission()
    # have it run around australia

    file_names = [
        "vege_720x360.csv",
        "topo_720x360.csv",
        "temp_720x360.csv",
        "rain_720x360.csv"
    ]

    images = readdlm.(maps_dir .* file_names, ',')

    for img in images
        img[img .== 99999] .= NaN
    end

    australia = (202:258, 587:668)

    lb = [0.0, 0.0]; ub = [1.0, 1.0]

    map0 = imgToMap(normalize(images[1][australia...]), lb, ub)
    sampler = MultiMap(map0)

    prior_maps = [imgToMap(normalize(img[australia...]), lb, ub) for img in images[2:end]]

    occupancy = imgToMap(Matrix{Bool}(isnan.(images[1][australia...])), lb, ub)


    ## initialize alg values
    weights = [1, 6, 1, 3e-3] # mean, std, dist, prox
    start_loc = [0.8, 0.6] # starting location
    num_samples = 50


    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations
    n = (5,5) # number of samples in each dimension
    axs_sp = range.(lb, ub, n)
    points_sp = vec(collect.(Iterators.product(axs_sp...)))
    prior_samples = [Sample((x, i+length(sampler)), d(x))
                     for (i, d) in enumerate(prior_maps)
                         for x in points_sp if !isnan(d(x))]

    # Calculate correlation coefficients
    [cor(map0.(points_sp), d.(points_sp)) for d in prior_maps]

    visualize(sampler.maps..., prior_maps...;
              titles=["Vegetation", "Elevation", "Ground Temperature", "Rainfall"],
              samples=points_sp)

    return Mission(; occupancy,
                       sampler,
                       start_loc,
                       weights,
                       num_samples,
                       prior_samples)

end

function conradMission()

    file_names = [
        "../maps/weightMap1.csv",
        "../maps/weightMap2.csv",
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

    sampler = MultiMap(maps)

    ## initialize alg values
    weights = [1, 6, 1, 3e-3] # mean, std, dist, prox
    start_loc = [0.8, 0.6] # starting location
    num_samples = 20

    occupancy = Map(zeros(Bool, n, n), lb, ub)

    return Mission(; occupancy,
                       sampler,
                       start_loc,
                       weights,
                       num_samples)
end

function rosMission()

    # this allows loading a module from within this function
    # it runs this block in the global namespace of this module
    # the benefit is that it only gets used if this function is run
    @eval begin
        # this requires a working rospy installation
        using ROSInterface: ROSConnection

        # NOTE switch these for swagbot nodes
        sub_nodes = [
            "/value1",
            "/value2"
        ]

        sampler = ROSConnection(sub_nodes)
    end

    lb = [0.0, 0.0]; ub = [1.0, 1.0]

    occupancy = Map(zeros(Bool, 100, 100), lb, ub)

    ## initialize alg values
    weights = [1, 6, 1, 3e-3] # mean, std, dist, prox
    start_loc = [0.0, 0.0]
    num_samples = 10

    return Mission(; occupancy,
                       sampler,
                       start_loc,
                       weights,
                       num_samples)
end

"""
$SIGNATURES

The main function that runs the adaptive sampling routine. For each iteration, a
sample location is selected, a sample is collected, the belief model is updated,
and visuals are possibly shown. The run finishes when the designated number of
samples is collected.

Inputs:

    - samples: a vector of samples, this can be used to jump-start a mission or
      resume a previous mission (default empty)
    - visuals: true or false to cause map plots to be shown or not (default false)
    - sleep_time: the amount of time to wait after each iteration, useful for
      visualizations (default 0)

Outputs:

    - samples: the new samples collected
    - beliefModel: the probabilistic representation of the quantities being
      searched for
"""
function (M::Mission)(; samples=Sample[], visuals=false, sleep_time=0)
    M.occupancy(M.start_loc) && error("start location is within obstacle")

    # initialize
    lb, ub = M.occupancy.lb, M.occupancy.ub
    new_loc = M.start_loc
    quantities = eachindex(M.sampler) # all current available quantities

    beliefModel = nothing
    if !isempty(samples)
        beliefModel = BeliefModel(samples, M.prior_samples, lb, ub)
    end
    sampleCost = nothing

    println("Mission started")
    println()

    for i in 1:M.num_samples
        println("Sample number $i")

        # new sample indices
        if beliefModel !== nothing # prior belief exists
            sampleCost = NormedSampleCost(M, samples, beliefModel, quantities)
            new_loc = selectSampleLocation(sampleCost, lb, ub)
            @debug "new location: $new_loc"
            @debug "cost function terms: $(Tuple(values(sampleCost, new_loc)) .* Tuple(M.weights))"
            @debug "cost function value: $(sampleCost(new_loc))"
        end

        # sample all quantities
        new_samples = takeSamples(new_loc, M.sampler)
        append!(samples, new_samples)

        # new belief
        beliefModel = BeliefModel(samples, M.prior_samples, lb, ub)

        # visualization
        if visuals
            display(visualize(M, beliefModel, sampleCost, samples, quantity=1))
        end
        @debug "output correlations: $(round.(outputCorMat(beliefModel), digits=3))"
        sleep(sleep_time)
    end

    println()
    println("Mission complete")
    return samples, beliefModel
end

end
