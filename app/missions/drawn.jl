
using Logging: global_logger, ConsoleLogger, Info, Debug
using DelimitedFiles: readdlm

using GridMaps: GridMap, pointToCell

using InformativeSampling
using .Samples: GridMapsSampler
using .SampleCosts: LogNormed
using .Missions: Mission

using InformativeSamplingUtils
using .DataIO: maps_dir

function drawnMission()

    file_names = [
        "weightMap1.csv",
        "weightMap2.csv",
    ]

    # need to drop the last point
    data = readdlm.(maps_dir .* file_names, ',')

    bounds = (
        lower = minimum(minimum(d, dims=1)[1:2] for d in data),
        upper = maximum(maximum(d, dims=1)[1:2] for d in data)
    )

    # scatter(data[1][:,1], data[1][:,2], marker_z=data[1][:,3])
    # scatter(data[2][:,1], data[2][:,2], marker_z=data[2][:,3])

    n = floor(Int, sqrt(maximum(size.(data, 1))))

    maps = [GridMap(zeros(n, n), bounds) for _ in 1:2]

    for (x,y,z) in eachrow(data[1])
        maps[1][pointToCell([x,y], maps[1])] = z
    end
    for (x,y,z) in eachrow(data[2])
        maps[2][pointToCell([x,y], maps[2])] = z
    end

    sampler = GridMapsSampler(maps)

    sampleCostType = LogNormed

    ## initialize alg values
    weights = [1, 6, 1, 3e-3] # mean, std, dist, prox
    start_locs = [[0.8, 0.6]] # starting locations
    num_samples = 20

    occupancy = GridMap(zeros(Bool, n, n), bounds)

    mission = Mission(;
        occupancy,
        sampler,
        num_samples,
        sampleCostType,
        weights,
        start_locs
    )

    return mission
end


#* Run

# set the logging level: Info or Debug
global_logger(ConsoleLogger(stderr, Info))

using .Visualization: vis

## initialize data for mission
mission = drawnMission()

## run search alg
@time samples, beliefs = mission(
    vis;
    sleep_time=0.0
);
