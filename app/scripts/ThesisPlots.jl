
using InformativeSampling
using .Maps, .Missions, .BeliefModels, .Samples, .ROSInterface, .Kernels

using InformativeSamplingUtils
using .DataIO

using Statistics: mean, std
using FileIO: load
using Plots
using Plots: mm
using Printf

# r is the range, contains the min and max values
function createColorbarTicks(r)
ticks = [ceil(r[1], sigdigits=3),
            round((r[2]-r[1])/2 + r[1], sigdigits=3),
            floor(r[2], sigdigits=3)]
return ticks, [@sprintf("%.1f", x) for x in ticks]
end

mkpath(output_dir * "thesis/aus_000")


#* load mission
dir = "new_aus/aus_multiKernel_zeromean_noises_fullpdf_nodrop_OnlyVar"
file_name = output_dir * "$dir/data_000" * output_ext

data = load(file_name)
maes = data["metrics"].mae
mxaes = data["metrics"].mxae
dists = cumsum(data["metrics"].dists)
times = cumsum(data["metrics"].times)

mission = data["mission"]
samples = data["samples"]
beliefs = data["beliefs"]

occ = mission.occupancy
quantities = eachindex(mission.sampler)
num_quant = length(mission.sampler)

xp = first.(getfield.(samples, :x))
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)

pred_range = (Inf, -Inf)
err_range = (Inf, -Inf)

for bm in beliefs
axs, points = generateAxes(occ)
pred, err = bm(tuple.(vec(points), 1))
global pred_range = (min(minimum(pred), pred_range[1]), max(maximum(pred), pred_range[2]))
global err_range = (min(minimum(err), err_range[1]), max(maximum(err), err_range[2]))
end

err_range = (0.0, err_range[2])

pred_ticks = createColorbarTicks(pred_range)
err_ticks = createColorbarTicks(err_range)

#* metrics

gr()

width, height = 1200, 800

p = plot(
[maes[1:30,:] mxaes[1:30,:]],
title="Prediction Errors",
xlabel="Sample Number",
ylabel="Prediction Error",
labels=["Mean Absolute Error" "Max Absolute Error"],
# seriescolors=[:black RGB(0.1,0.7,0.2)],
framestyle=:box,
marker=true,
ylim=(0,1),
titlefontsize=24,
markersize=8,
tickfontsize=15,
labelfontsize=20,
legendfontsize=16,
margin=5mm,
linewidth=4,
size=(width, height)
)
# p |> display
savefig(output_dir * "thesis/aus_000/errors.png")

p = plot(
dists[1:30,:],
title="Distance Traveled",
xlabel="Sample Number",
ylabel="Cumulative Distance",
labels=false,
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
# p |> display
savefig(output_dir * "thesis/aus_000/distances.png")


#* ground truth

pyplot()

axs, _ = generateAxes(mission.sampler[1])

p = heatmap(axs..., mission.sampler[1]';
title="Vegetation Ground Truth",
framestyle=:none,
titlefontsize=21,
colorbar_tickfontsize=17,
)
# display(p)
savefig(output_dir * "thesis/aus_000/ground_truth.png")


#* frames

pyplot()

mkpath(output_dir * "thesis/aus_000/belief_frames")
mkpath(output_dir * "thesis/aus_000/uncertainty_frames")

for (i, bm) in enumerate(beliefs)
axs, points = generateAxes(occ)
pred_map, err_map = bm(tuple.(points, 1))

pred_map[occ] .= NaN
err_map[occ] .= NaN

p1 = heatmap(axs..., pred_map';
    title="Predicted Values",
    framestyle=:none,
    titlefontsize=21,
    colorbar_tickfontsize=17,
    clim=(0, 1),
)
scatter!(x1[1:i], x2[1:i];
    label=false,
    color=:green,
    markersize=8)
savefig(output_dir * "thesis/aus_000/belief_frames/$(lpad(i,2,'0')).png")

p2 = heatmap(axs..., err_map';
    title="Uncertainties",
    framestyle=:none,
    titlefontsize=21,
    colorbar_tickfontsize=17,
    # clim=err_range,
    # colorbar_ticks=err_ticks,
)
scatter!(x1[1:i], x2[1:i];
    label=false,
    color=:green,
    markersize=8)
savefig(output_dir * "thesis/aus_000/uncertainty_frames/$(lpad(i,2,'0')).png")
end

#* figure

pyplot()

axs, _ = generateAxes(mission.sampler[1])

p01 = heatmap(axs..., mission.sampler[1]';
    title="Ground Truth",
    framestyle=:none,
    titlefontsize=19,
    colorbar_tickfontsize=17,
)

p0r = heatmap(axs..., mission.sampler[1]';
    title="",
    framestyle=:none,
    titlefontsize=19,
    colorbar_tickfontsize=17,
)

i=5
plots = map(5:5:length(beliefs)) do i
    bm = beliefs[i]

    axs, points = generateAxes(occ)
    pred_map, err_map = bm(tuple.(points, 1))

    pred_map[occ] .= NaN
    err_map[occ] .= NaN

    pred_title = i==5 ? "Predicted Values" : ""
    p1 = heatmap(axs..., pred_map';
        title=pred_title,
        framestyle=:none,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=(0, 1),
    )
    scatter!(xp[1:i], x2[1:i];
        label=false,
        color=:green,
        markersize=8)

    err_title = i==5 ? "Uncertainties" : ""
    p2 = heatmap(axs..., err_map';
        title=err_title,
        framestyle=:none,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        # clim=err_range,
        # colorbar_ticks=err_ticks,
        right_margin=-5mm,
    )
    scatter!(x1[1:i], x2[1:i];
        label=false,
        color=:green,
        markersize=8)

    return (i==5 ? p01 : p0r), p1, p2
end

p = plot(Iterators.flatten(plots)...,
    layout=(6, 3),
    size=(1000, 1300))

savefig(output_dir * "thesis/aus_000/full_run_comparison.png")
