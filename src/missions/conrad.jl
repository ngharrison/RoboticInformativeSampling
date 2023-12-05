using DelimitedFiles: readdlm

using AdaptiveSampling: Maps, Samples, SampleCosts, Missions, Visualization

using .Maps: Map, pointToCell
using .Samples: MapsSampler
using .SampleCosts: NormedSampleCost
using .Missions: Mission

function conradMission()

    file_names = [
        "weightMap1.csv",
        "weightMap2.csv",
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
    start_locs = [[0.8, 0.6]] # starting locations
    num_samples = 20

    occupancy = Map(zeros(Bool, n, n), lb, ub)

    return Mission(; occupancy,
                   sampler,
                   num_samples,
                   sampleCostType,
                   weights,
                   start_locs)
end

