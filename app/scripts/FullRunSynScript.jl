#* packages and functions

using MultiQuantityGPs
using MultiQuantityGPs.Kernels
using GridMaps

using InformativeSampling
using .Missions, .Samples

include(dirname(Base.active_project()) * "/ros/ROSInterface.jl")
using .ROSInterface

using InformativeSamplingUtils
using .DataIO

using LinearAlgebra: norm
using Statistics: mean, std, cor, quantile
using FileIO: load
using Plots
using Plots: mm
using Printf
using DelimitedFiles: readdlm

# r is the range, contains the min and max values
function createColorbarTicks((l, h))
    ticks = [ceil(l, sigdigits=3),
        round((h - l) / 2 + l, sigdigits=3),
        floor(h, sigdigits=3)]
    return ticks, [@sprintf("%.1g", x) for x in ticks]
end

num_samples = 30


#* load mission

runs = [
    "syn_mtoKernel_means_noises_fullpdf_nodrop_DistScaledEIGF_alpha5",
    "syn_multiKernel_means_noises_condpdf_nodrop_DistScaledEIGF_alpha5",
    "syn_multiKernel_means_noises_fullpdf_hypdrop_DistScaledEIGF_alpha5",
    "syn_multiKernel_means_noises_fullpdf_nodrop_DerivVar_alpha50",
    "syn_multiKernel_means_noises_fullpdf_nodrop_DistScaledEIGF_alpha5",
    "syn_multiKernel_means_noises_fullpdf_nodrop_EIGF_alpha5",
    "syn_multiKernel_means_noises_fullpdf_nodrop_OnlyVar_alpha5",
    "syn_multiKernel_zeromean_noises_fullpdf_nodrop_OnlyVar_alpha5",
    "syn_multiKernel_means_noises_fullpdf_nodrop_DistEIGF1e-2_alpha5",
]

run = runs[parse(Int, ARGS[1])]

priors = "111"
# extra = run[5:7] == "mto" ? "_fix_mto_2" : ""
file_name = output_dir * "syn_alphas/$(run)/data_$(priors)" * output_ext

save_file = output_dir * "thesis/syn_$(priors)_full_runs/$(run)_paths.png"
mkpath(save_file[1:findlast('/', save_file)])

data = load(file_name)
maes = data["metrics"][end-2].mae
mxaes = data["metrics"][end-2].mxae
dists = cumsum(data["metrics"][end-2].dists)
times = cumsum(data["metrics"][end-2].times)

mission = data["missions"][end-2].mission
samples = data["missions"][end-2].samples
beliefs = data["missions"][end-2].beliefs

occ = mission.occupancy
quantities = eachindex(mission.sampler)
num_quant = length(mission.sampler)

xp = first.(getfield.(samples, :x))
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)

pred_range = (Inf, -Inf)
err_range = (Inf, -Inf)

for bm in @view beliefs[5:5:end]
    axs, points = generateAxes(occ)
    pred_map, err_map = bm(tuple.(vec(points), 1))
    mask = vec(.! mission.occupancy)

    global pred_range = (min(minimum(pred_map[mask]), pred_range[1]),
                         max(maximum(pred_map[mask]), pred_range[2]))
    global err_range = (min(minimum(err_map[mask]), err_range[1]),
                        max(maximum(err_map[mask]), err_range[2]))
end

err_range = (0.0, err_range[2])

pred_ticks = createColorbarTicks(pred_range)
err_ticks = createColorbarTicks(err_range)

function addPathsAndMarkers(i)
    # to current plot
    # plot the paths
    for j in 1:i-1
        # path = cellToPoint.(paths_cache[j], Ref(occ))
        path = xp[j:j+1]
        plot!(first.(path), last.(path);
            label=false, color=:gray, line=:dash, lineopacity=0.7)
    end
    if i < length(samples)
        # path = cellToPoint.(paths_cache[i], Ref(occ))
        path = xp[i:i+1]
        plot!(first.(path), last.(path);
            label=false, color=:gray, line=:dash, linewidth=2)
    end

    # plot the markers
    scatter!(x1[begin:i-1], x2[begin:i-1];
        label=false,
        color=:green,
        markersize=6)
    scatter!(x1[i:i], x2[i:i];
        label=false,
        color=:royalblue,
        shape=:utriangle,
        markersize=12)
    if i < length(samples)
        scatter!(x1[i+1:i+1], x2[i+1:i+1],
            label=false,
            color=:red,
            shape=:xcross,
            markersize=10)
    end
end

#* full run comparison

try
    pyplot()
catch
    pyplot()
end

i=5
plots = map(5:5:length(beliefs)) do i
    axs, _ = generateAxes(mission.sampler[1])

    gt_title = i == 5 ? "Ground Truth" : ""
    p0 = heatmap(axs..., mission.sampler[1]';
        title=gt_title,
        ylabel="$i Samples",
        framestyle=:none,
        # aspect_ratio=:equal,
        right_margin=-10mm,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        labelfontsize=17,
    )

    bm = beliefs[i]

    axs, points = generateAxes(occ)
    pred_map, err_map = bm(tuple.(points, 1))

    pred_map[occ] .= NaN
    err_map[occ] .= NaN

    pred_title = i == 5 ? "Predicted Values" : ""
    p1 = heatmap(axs..., pred_map';
        title=pred_title,
        framestyle=:none,
        # aspect_ratio=:equal,
        right_margin=-10mm,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=(0, 1),
    )
    addPathsAndMarkers(i)

    err_title = i == 5 ? "Uncertainties" : ""
    p2 = heatmap(axs..., err_map';
        title=err_title,
        framestyle=:none,
        # aspect_ratio=:equal,
        right_margin=-10mm,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=err_range,
        # colorbar_ticks=err_ticks,
    )
    addPathsAndMarkers(i)

    sampleCost = mission.sampleCostType(
        occ, samples[1:i], bm, quantities, mission.weights
    )
    obj_map = -sampleCost.(points)

    lessNotInf = x -> x === -Inf ? Inf : x
    greaterNotInf = x -> x === Inf ? -Inf : x

    obj_range = (0.0, maximum(greaterNotInf, obj_map))
    obj_ticks = createColorbarTicks(obj_range)

    obj_title = i == 5 ? "Sample Utility" : ""
    p3 = heatmap(axs..., obj_map';
        title=obj_title,
        framestyle=:none,
        # aspect_ratio=:equal,
        right_margin=-10mm,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=obj_range,
        colorbar_ticks=obj_ticks,
    )
    addPathsAndMarkers(i)

    return p0, p1, p2, p3
end

p = plot(Iterators.flatten(plots)...,
    layout=(6, 4),
    size=(1100, 1300))

savefig(save_file)
