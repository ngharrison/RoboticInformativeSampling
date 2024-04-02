# script for analyzing data from missions
# can be run after Main.jl or by opening a saved file

using AdaptiveSampling

using .Missions: Mission
using .BeliefModels: BeliefModel
using .Samples: Sample
using .Outputs: output_dir, output_ext

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


## Calculating statistical significance of difference between means
using Distributions
p = 4; a = 8; b = 2; n = 18;
t = abs(err_means[p,a] - err_means[p,b])/sqrt(err_stds[p,a]^2/n + err_stds[p,b]^2/n)
cdf(TDist(n-1), t) # t-test

# calc number of samples from t-score
t^2*(err_stds[p,a]^2 + err_stds[p,b]^2)/(err_means[p,a] - err_means[p,b])^2


## Aus
dir = "aus_ave_means_noise"
# file_name = output_dir * "2023-08-28-16-58-29_metrics" * output_ext
file_name_s = output_dir * "$dir/metrics_000" * output_ext
file_name_m = output_dir * "$dir/metrics_111" * output_ext

data = load(file_name_s)
maes = data["metrics"].mae

data = load(file_name_m)
maes = [maes data["metrics"].mae]
cors = data["metrics"].cors
dets = [u.^2 for u in cors]

width, height = 1200, 800

plot(
    hcat((c[2:end] for c in cors[1:30,:])...)',
    title="Estimated Correlation to Vegetation",
    labels=["Elevation" "Ground Temperature" "Rainfall"],
    xlabel="Sample Number",
    ylabel="Correlation",
    seriescolors=[RGB(0.4, 0.2, 0.1) RGB(0.9, 0.4, 0.0) RGB(0.1,0.3,1)],
    legend=(0.73,0.42),
    framestyle=:box,
    marker=true,
    ylim=(-1,1),
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

savefig(output_dir * "$dir/$(dir)_correlations.png")

p = plot(
    maes[1:30,:],
    title="Prediction Errors",
    xlabel="Sample Number",
    ylabel="Mean Absolute Map Error",
    labels=["No Priors" "Priors"],
    seriescolors=[:black RGB(0.1,0.7,0.2)],
    framestyle=:box,
    marker=true,
    # ylim=(0,.5),
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

savefig(output_dir * "$dir/$(dir)_errors.png")

## Batch
dir = "batch_means_noise_1e2"

p_err = Vector{Any}(undef, 8)
p_cor = Vector{Any}(undef, 8)

err_means = zeros(30, 8)
err_stds = zeros(30, 8)

det_means = Vector{Any}(undef, 8)
det_stds = Vector{Any}(undef, 8)

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
    cors = [run.cors for run in data["metrics"]]
    dets = [[v.^2 for v in u] for u in cors]

    err_means[:,i] .= mean(maes)
    err_stds[:,i] .= std(maes)

    det_means[i] = mean(dets)
    det_stds[i] = stdM(dets)

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
    ylim=(0,.3),
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


## True mean coefficients
mission_peaks = [3,3,4,4,5,5]
cors = Array{Any, 1}(undef, length(mission_peaks))
for (i, num_peaks) in enumerate(mission_peaks)
    mission, c = simMission(; seed_val=i, num_peaks)
    cors[i] = c
end
s = [abs.(c) for c in cors]
mean(s)
std(s)

# From the map abs correlations:
#
# julia> mean(s)
# 3-element Vector{Float64}:
#  0.9999999999999996
#  0.7786774403828388
#  0.21356037266496505
#
# julia> std(s)
# 3-element Vector{Float64}:
#  2.5316980181136773e-16
#  0.01919544968657839
#  0.13245078231003993

# Determination coefficients:
#
# julia> mean(s)
# 3-element Vector{Float64}:
#  0.9999999999999991
#  0.6066456105683947
#  0.060227374218250064
#
# julia> std(s)
# 3-element Vector{Float64}:
#  5.063396036227355e-16
#  0.029810722128705513
#  0.05812149041402016
