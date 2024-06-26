
using Logging: global_logger, ConsoleLogger, Info, Debug
using LinearAlgebra: I, norm
using Statistics: mean, cor
using Random: seed!

using InformativeSampling
using .Maps: Map, generateAxes
using .Samples: Sample, MapsSampler
using .SampleCosts: MIPT, EIGF, DistScaledEIGF, DerivVar, DistScaledDerivVar
using .Missions: Mission
using .Kernels: mtoKernel

using InformativeSamplingUtils
using .DataIO: GaussGroundTruth, Peak
using .Visualization: vis

function simMission(; seed_val=0, num_samples=30, num_peaks=3, priors=Bool[1,1,1])
    seed!(seed_val) # make random values deterministic

    bounds = (lower = [0.0, 0.0], upper = [1.0, 1.0])

    occupancy = Map(zeros(Bool, 100, 100), bounds)

    ## initialize ground truth

    # simulated
    peaks = map(1:num_peaks) do _
        μ = rand(2).*(bounds.upper-bounds.lower) .+ bounds.lower
        Σ = 0.02*(rand()+0.5)*mean(bounds.upper-bounds.lower)^2*I
        h = rand()
        Peak(μ, Σ, h)
    end
    ggt = GaussGroundTruth(peaks)
    _, points = generateAxes(occupancy)
    mat = ggt(points)
    map0 = Map(mat./maximum(mat), bounds)

    ## Create prior prior_samples

    # none -- leave uncommented
    prior_maps = []

    # multiplicative
    m = Map(abs.(map0 .* randn()), bounds)
    push!(prior_maps, m)

    # additive
    m = Map(abs.(map0 .+ 0.2 .* randn(size(map0))), bounds)
    push!(prior_maps, m)

    # # both
    # push!(prior_maps, Map(abs.(map0 .* randn() + 0.1 .* randn(size(map0))), bounds))

    # # spatial shift
    # t = rand(1:7)
    # push!(prior_maps, [zeros(size(map0,1),t) map0[:,1:end-t]]) # shift

    # random peaks
    peaks = map(1:num_peaks) do _
        μ = rand(2).*(bounds.upper-bounds.lower) .+ bounds.lower
        Σ = 0.02*(rand()+0.5)*mean(bounds.upper-bounds.lower)^2*I
        h = rand()
        Peak(μ, Σ, h)
    end
    tggt = GaussGroundTruth(peaks)
    tmat = tggt(points)
    m = Map(tmat./maximum(tmat), bounds)
    push!(prior_maps, m)

    # # purely random values
    # m = Map(rand(size(map0)...), bounds)
    # push!(prior_maps, m)

    sampler = MapsSampler(map0)

    sampleCostType = DistScaledEIGF

    ## initialize alg values
    # weights = (; μ=17, σ=1.5, τ=7)
    # weights = (; μ=3, σ=1, τ=.5, d=1)
    # weights = (; μ=1, σ=1e1, τ=1, d=1) # sogp
    weights = (; μ=1, σ=1e2, τ=1, d=1) # others
    # weights = (; μ=1, σ=1, τ=.1, d=1)
    start_locs = [] # starting location

    # n = (4,4) # number of samples in each dimension
    # axs_sp = range.(bounds..., n)
    # start_locs = vec(collect.(Iterators.product(axs_sp...))) # starting locations


    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations
    n = (5,5) # number of samples in each dimension
    axs_sp = range.(bounds..., n)
    points_sp = vec(collect.(Iterators.product(axs_sp...)))
    prior_samples = [Sample((x, i+length(sampler)), d(x))
                     for (i, d) in enumerate(prior_maps[priors])
                         for x in points_sp if !isnan(d(x))]

    # Calculate correlation coefficients
    @debug [cor(map0.(points_sp), d.(points_sp)) for d in prior_maps]
    # @debug [cor(vec(map0), vec(d)) for d in prior_maps]

    noise = (value=0.0, learned=true)

    mission = Mission(;
        occupancy,
        sampler,
        num_samples,
        sampleCostType,
        weights,
        start_locs,
        prior_samples,
        noise
    )

    return mission, prior_maps

end


#* Run

# set the logging level: Info or Debug
global_logger(ConsoleLogger(stderr, Info))

## initialize data for mission
mission, prior_maps = simMission(num_samples=10)

vis(mission.sampler..., prior_maps...;
    titles=["QOI", "Scaling Factor", "Additive Noise", "Random Map"],
    points=first.(getfield.(mission.prior_samples, :x)))

## run search alg
@time samples, beliefs, times = mission(
    vis;
    sleep_time=0.0
);


# using .Metrics: calcMetrics
# using .DataIO: save
#
# ## calculate errors
# metrics = calcMetrics(mission, samples, beliefs, times, 1)
#
# ## save outputs
# save(mission, samples, beliefs; animation=true)
# save(metrics)


#* Batch

# set the logging level: Info or Debug
global_logger(ConsoleLogger(stderr, Info))

using .BeliefModels: outputCorMat
using .Metrics: calcMetrics
using .DataIO: save

mission, _ = simMission()
dir = "batch_means_eigf_noise"
save(mission, [], []; file_name="$(dir)/mission")

mission_peaks = [3,3,4,4,5,5]
num_runs = 3
metrics = Array{Any, 2}(undef, (length(mission_peaks), num_runs))
# pick all the prior data combinations
@time for priors in Iterators.product(fill(0:1,3)...)
    for (i, num_peaks) in enumerate(mission_peaks)
        ## initialize data for mission
        mission, _ = simMission(; seed_val=i, num_peaks, priors=collect(Bool, priors))
        for j in 1:num_runs
            println()
            println("Priors ", priors)
            println("Peaks number ", i)
            println("Run number ", j)
            println()
            ## run search alg
            @time samples, beliefs, times = mission(seed_val=j, sleep_time=0.0);
            @debug "output correlation matrix:" outputCorMat(beliefs[end])
            # save(mission, samples, beliefs; animation=true)
            ## calculate errors
            metrics[i,j] = calcMetrics(mission, samples, beliefs, times, 1)
        end
    end
    ## save outputs
    save(metrics; file_name="$(dir)/metrics_$(join(priors))")
end

#* Data analysis

using InformativeSampling

using InformativeSamplingUtils
using .DataIO: output_dir, output_ext

using Statistics: mean, std
using FileIO: load
using Plots
using Plots: mm

"""
Calculates the standard deviation across elements of an array when the elements
are arrays of arrays.
"""
function stdM(cors)
    d = cors .- Ref(mean(cors))
    l = [[v.^2 for v in u] for u in d]
    s = sum(l) / (length(cors) - 1)
    return [[sqrt(v) for v in u] for u in s]
end


p_err = Vector{Any}(undef, 8)
p_cor = Vector{Any}(undef, 8)

err_means = zeros(30, 8)
err_stds = zeros(30, 8)

max_err_means = zeros(30, 8)
max_err_stds = zeros(30, 8)

det_means = Vector{Any}(undef, 8)
det_stds = Vector{Any}(undef, 8)

dist_means = zeros(30, 8)
dist_stds = zeros(30, 8)

time_means = zeros(30, 8)
time_stds = zeros(30, 8)

priors = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1)
]
chars = "HML"
colors = [(1,0,0), (0,1,0), (0,0,1)]

for (i, p) in enumerate(priors)
    file_name = output_dir * "$dir/metrics_$(join(p))" * output_ext

    data = load(file_name)
    maes = [run.mae for run in data["metrics"]]
    mxaes = [run.mxae for run in data["metrics"]]
    cors = [run.cors for run in data["metrics"]]
    dets = [[v.^2 for v in u] for u in cors]
    dists = [cumsum(run.dists) for run in data["metrics"]]
    times = [cumsum(run.times) for run in data["metrics"]]

    err_means[:,i] .= mean(maes)
    err_stds[:,i] .= std(maes)

    max_err_means[:,i] .= mean(mxaes)
    max_err_stds[:,i] .= std(mxaes)

    det_means[i] = mean(dets)
    det_stds[i] = stdM(dets)

    dist_means[:,i] .= mean(dists)
    dist_stds[:,i] .= std(dists)

    time_means[:,i] .= mean(times)
    time_stds[:,i] .= std(times)

    plt = plot(hcat((c[2:end] for c in mean(dets))...)',
               ribbon=hcat((c[2:end] for c in stdM(dets))...)',
               framestyle=:box,
               labels=false,
               markers=true,
               left_margin=(i∈(2,4,6,8) ? 5mm : 0mm),
               bottom_margin=(i∈(7,8) ? 5mm : 0mm),
               xticks=(i∈(7,8) ? (0:5:30) : false),
               yticks=(i∈(2,4,6,8) ? (0:0.5:1) : false),
               xlabel=(i∈(7,8) ? "Sample Number" : ""),
               # ylabel=(i∈(2,4,6,8) ? "\$ \\mathbf{\\rho^2} \$" : ""),
               seriescolors=[(RGB((c.*0.8)...) for (i, c) in zip(p, colors) if i==1)...;;])
    p_cor[i] = plt
end

width, height = 1200, 800

p_cor[1] = plot([2 2 2],
                framestyle=:none,
                legend=(0.4, 0.7),
                linewidth=4,
                xticks=false,
                yticks=false,
                markers=false,
                legendfontsize=16,
                labels=[" High" " Medium" " Low"],
                seriescolors=[(RGB((c.*0.8)...) for c in colors)...;;])
plot(
    p_cor[[2,1,4,3,6,5,8,7]]...,
    plot_title="Hypothesis Scores",
    plot_titlefontsize=24,
    ylim=(0,1),
    grid=false,
    tickfontsize=15,
    labelfontsize=20,
    markersize=6,
    # legendcolumns=3,
    fg_legend=:transparent,
    layout=grid(4,2),
    # layout=@layout([grid(3,2){0.75h}; _ a{0.5w} b]),
    size=(width, height)
)
gui()

savefig(output_dir * "$dir/hypothesis_scores.png")

plot(
    err_means,
    # ribbon=err_stds,
    xlabel="Sample Number",
    ylabel="Mean Absolute Map Error",
    title="Prediction Errors",
    ylim=(0,.35),
    seriescolors=[(RGB((p.*0.8)...) for p in priors)...;;],
    labels=[replace([join(c for (p, c) in zip(p, chars) if p==1) for p in priors], ""=>"none")...;;],
    framestyle=:box,
    markers=true,
    legendcolumns=2, # OR layout=2,
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
gui()

savefig(output_dir * "$dir/errors.png")

plot(
    max_err_means,
    # ribbon=max_err_stds,
    xlabel="Sample Number",
    ylabel="Max Absolute Map Error",
    title="Max Prediction Errors",
    ylim=(0,1),
    seriescolors=[(RGB((p.*0.8)...) for p in priors)...;;],
    labels=[replace([join(c for (p, c) in zip(p, chars) if p==1) for p in priors], ""=>"none")...;;],
    framestyle=:box,
    markers=true,
    legendcolumns=2, # OR layout=2,
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
gui()

savefig(output_dir * "$dir/max_errors.png")

plot(
    dist_means,
    # ribbon=err_stds,
    xlabel="Sample Number",
    ylabel="Cumulative Distance",
    title="Distance Traveled",
    ylim=(0,22),
    seriescolors=[(RGB((p.*0.8)...) for p in priors)...;;],
    labels=[replace([join(c for (p, c) in zip(p, chars) if p==1) for p in priors], ""=>"none")...;;],
    framestyle=:box,
    markers=true,
    legendcolumns=2, # OR layout=2,
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    legend=:topleft,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
gui()

savefig(output_dir * "$dir/distances.png")

idxs = [1,2,3,5,4,6,7,8]
bar(
    replace([join(c for (p, c) in zip(p, chars) if p==1) for p in priors][idxs], ""=>"none"),
    time_means[end,idxs]/size(time_means,1),
    xlabel="Sample Number",
    ylabel="Average Computation Time (s)",
    title="Computation Time per Sample",
    ylim=(0,2),
    seriescolors=[RGB((p.*0.8)...) for p in priors][idxs],
    framestyle=:box,
    markers=true,
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legend=nothing,
    margin=5mm,
    size=(width, height)
)
gui()

savefig(output_dir * "$dir/computation_times.png")
