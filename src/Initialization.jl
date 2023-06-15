
module Initialization


using LinearAlgebra: I
using Images: load, imresize, Gray, gray
using DelimitedFiles: readdlm
using Statistics: cor
using Random: seed!

using Environment: Map, imgToMap, GaussGroundTruth, MultiMap, Peak, pointToCell
using Samples: Sample
using Visualization: visualize

"""
Inputs:
    - occupancy: an occupancy map, true in cells that are occupied
    - groundTruth: a function that returns a measurement value at any point
    - start_loc: the starting location
    - weights: weights for picking the next sample location
    - num_samples: the number of samples to collect in one run (default 20)
    - prior_samples: any samples taken previously (default empty)
    - samples: any samples taken already during the current mission (default empty)
"""
@kwdef struct MissionData
    occupancy
    groundTruth
    start_loc
    weights
    num_samples
    prior_samples = Sample[]
    samples = Sample[]
end

function simData()
    seed!(2) # make random values deterministic

    lb = [0, 0]; ub = [1, 1]

    # read in elevation
    elev_img = load("maps/arthursleigh_shed_small.tif")
    elevMap = imgToMap(gray.(elev_img), lb, ub)

    # read in obstacles
    obs_img = load("maps/obstacles_fieldsouth_220727.tif")
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
    subGroundTruth = Map(ggt(points), lb, ub)

    ## Create prior prior_samples

    # none -- leave uncommented
    prior_maps = []

    # additive
    push!(prior_maps, Map(abs.(subGroundTruth .+ 0.1 .* randn(size(subGroundTruth))), lb, ub))

    # multiplicative
    push!(prior_maps, Map(abs.(subGroundTruth .* randn()), lb, ub))

    # # both
    # push!(prior_maps, Map(abs.(subGroundTruth .* randn() + 0.1 .* randn(size(subGroundTruth))), lb, ub))

    # # spatial shift
    # t = rand(1:7)
    # push!(prior_maps, [zeros(size(subGroundTruth,1),t) subGroundTruth[:,1:end-t]]) # shift

    # purely random
    num_peaks = 3
    peaks = [Peak(rand(2).*(ub-lb) .+ lb, 0.02*I, rand())
             for i in 1:num_peaks]
    tggt = GaussGroundTruth(peaks)
    # push!(prior_maps, Map(tggt(points), lb, ub))

    groundTruth = MultiMap(subGroundTruth, Map(tggt(points), lb, ub))


    ## initialize alg values
    weights = [1, 6, 1, 1e-2] # mean, std, dist, prox
    start_loc = [0.5, 0.2] # starting location
    num_samples = 20


    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations
    n = (5,5) # number of samples in each dimension
    axs_sp = range.(lb, ub, n)
    points_sp = vec(collect.(Iterators.product(axs_sp...)))
    prior_samples = [Sample((x, i+length(groundTruth)), d(x))
                     for (i, d) in enumerate(prior_maps)
                         for x in points_sp if !isnan(d(x))]

    # Calculate correlation coefficients
    [cor(subGroundTruth.(points_sp), d.(points_sp)) for d in prior_maps]

    visualize(groundTruth.maps..., prior_maps...;
              titles=["Ground Truth 1", "Ground Truth 2", "Prior 1", "Prior 2"],
              samples=points_sp)

    return MissionData(; occupancy,
                       groundTruth,
                       start_loc,
                       weights,
                       num_samples,
                       prior_samples)

end


normalize(a) = a ./ maximum(filter(!isnan, a))


function realData()
    # have it run around australia

    file_names = [
        "maps/vege_720x360.csv",
        "maps/topo_720x360.csv",
        "maps/temp_720x360.csv",
        "maps/rain_720x360.csv"
    ]

    images = readdlm.(file_names, ',')

    for img in images
        img[img .== 99999] .= NaN
    end

    australia = (202:258, 587:668)

    lb = [0, 0]; ub = [1, 1]

    subGroundTruth = imgToMap(normalize(images[1][australia...]), lb, ub)
    groundTruth = MultiMap(subGroundTruth)

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
    prior_samples = [Sample((x, i+length(groundTruth)), d(x))
                     for (i, d) in enumerate(prior_maps)
                         for x in points_sp if !isnan(d(x))]

    # Calculate correlation coefficients
    [cor(subGroundTruth.(points_sp), d.(points_sp)) for d in prior_maps]

    visualize(groundTruth.maps..., prior_maps...;
              titles=["Vegetation", "Elevation", "Ground Temperature", "Rainfall"],
              samples=points_sp)

    return MissionData(; occupancy,
                       groundTruth,
                       start_loc,
                       weights,
                       num_samples,
                       prior_samples)

end

function conradData()

    file_names = [
        "maps/weightMap1.csv",
        "maps/weightMap2.csv",
    ]

    # need to drop the last point
    data = readdlm.(file_names, ',')

    lb = minimum(minimum(d, dims=1)[1:2] for d in data)
    ub = maximum(maximum(d, dims=1)[1:2] for d in data)

    # scatter(data[1][:,1], data[1][:,2], marker_z=data[1][:,3])
    # scatter(data[2][:,1], data[2][:,2], marker_z=data[2][:,3])

    n = floor(Int, sqrt(maximum(size.(data, 1))))

    groundTruths = [Map(zeros(n, n), lb, ub) for _ in 1:2]

    for (x,y,z) in eachrow(data[1])
        groundTruths[1][pointToCell([x,y], groundTruths[1])] = z
    end
    for (x,y,z) in eachrow(data[2])
        groundTruths[2][pointToCell([x,y], groundTruths[2])] = z
    end

    groundTruth = MultiMap(groundTruths)

    ## initialize alg values
    weights = [1, 6, 1, 3e-3] # mean, std, dist, prox
    start_loc = [0.8, 0.6] # starting location
    num_samples = 20

    occupancy = Map(zeros(Bool, n, n), lb, ub)

    return MissionData(; occupancy,
                       groundTruth,
                       start_loc,
                       weights,
                       num_samples)
end

function rosData()

    # this allows loading a module from within this function
    # it runs this block in the global namespace of this module
    # the benefit is that it only gets used if this function is run
    @eval begin
        # this requires a working rospy installation
        using ROSInterface: ROSConnection

        # TODO switch these for swagbot nodes
        sub_nodes = [
            "/value1",
            "/value2"
        ]

        groundTruth = ROSConnection(sub_nodes)
    end

    lb = [0, 0]; ub = [1, 1]

    occupancy = Map(zeros(Bool, 100, 100), lb, ub)

    ## initialize alg values
    weights = [1, 6, 1, 3e-3] # mean, std, dist, prox
    start_loc = [0.0, 0.0]
    num_samples = 10

    return MissionData(; occupancy,
                       groundTruth,
                       start_loc,
                       weights,
                       num_samples)
end

end
