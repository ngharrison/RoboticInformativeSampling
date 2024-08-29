
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

mkpath(output_dir * "thesis/syn_000")


#* load mission
dir = "new_syn/syn_multiKernel_zeromean_noises_fullpdf_nodrop_OnlyVar"
file_name = output_dir * "$dir/data_000" * output_ext

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
ylim=(0,1.2),
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
savefig(output_dir * "thesis/syn_000/errors.png")

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
savefig(output_dir * "thesis/syn_000/distances.png")


#* ground truth

pyplot()

axs, _ = generateAxes(mission.sampler[1])

p = heatmap(axs..., mission.sampler[1]';
title="Vegetation Ground Truth",
framestyle=:none,
titlefontsize=21,
colorbar_tickfontsize=17,
)
display(p)
savefig(output_dir * "thesis/syn_000/ground_truth.png")


#* frames

pyplot()

mkpath(output_dir * "thesis/syn_000/belief_frames")
mkpath(output_dir * "thesis/syn_000/uncertainty_frames")

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
savefig(output_dir * "thesis/syn_000/belief_frames/$(lpad(i,2,'0')).png")

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
savefig(output_dir * "thesis/syn_000/uncertainty_frames/$(lpad(i,2,'0')).png")
end

#* full run comparison

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
    scatter!(x1[1:i], x2[1:i];
        label=false,
        color=:green,
        markersize=8)

    err_title = i==5 ? "Uncertainties" : ""
    p2 = heatmap(axs..., err_map';
        title=err_title,
        framestyle=:none,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=err_range,
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

savefig(output_dir * "thesis/syn_000/full_run_comparison.png")

#* distance scaling for objective function

using Plots

# gradually-delayed distance scaling

n_scale = n -> 2/(1 + exp(1 - n)) - 1
plot(
    n_scale, 1, 8,
    title="Sample Scaling",
    xlabel="Sample Number",
    ylabel="Scaling Factor",
    framestyle=:box,
    legend=false,
    titlefontsize=19,
    tickfontsize=13,
    labelfontsize=17,
    linewidth=3,
    size=(1050,750),
)|>display

savefig(output_dir * "thesis/sample_scaling.png")

d_scale = d -> 1/(1 + 1*d^2)
plot(
    d_scale, 0, 2,
    title="Distance Scaling",
    xlabel="Distance",
    ylabel="Scaling Factor",
    framestyle=:box,
    legend=false,
    titlefontsize=19,
    tickfontsize=13,
    labelfontsize=17,
    linewidth=3,
    size=(1050,750),
)|>display

savefig(output_dir * "thesis/distance_scaling.png")

full_scale = (n,d) -> 1/(1 + n_scale(n)*d^2)
n, d = 1:8, 0:0.01:2
# z = Surface((x,y)->rosenbrock([x,y]), x, y)
surface(
    n, d, full_scale,
    xlabel="Samples",
    ylabel="Distance",
    zlabel="Scaling",
    camera = (60, 20)
)|>display

#* load mission
dir = "new_syn/syn_multiKernel_zeromean_noises_fullpdf_nodrop_OnlyVar"
file_name = output_dir * "$dir/data_000" * output_ext

data = load(file_name)
maes = data["metrics"][1].mae
mxaes = data["metrics"][1].mxae
dists = cumsum(data["metrics"][1].dists)
times = cumsum(data["metrics"][1].times)

mission = data["missions"][1].mission
samples = data["missions"][1].samples
beliefs = data["missions"][1].beliefs

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

#* objective function plots

using .SampleCosts, .Samples, .Maps

pyplot()

# mean, var, derivvar, eigf

i = 20

bm = beliefs[i]
occ = mission.occupancy

axs, points = generateAxes(occ)
pred_map, err_map = bm(tuple.(points, 1))

# blocked points
pred_map[occ] .= NaN
err_map[occ] .= NaN

pred_plt = heatmap(axs..., pred_map',
    title="GP Mean",
    framestyle=:none,
    titlefontsize=26,
    colorbar_tickfontsize=18,
    clim=(0, 1),
)
scatter!(x1[1:i], x2[1:i];
    label=false,
    color=:green,
    markersize=8)

err_plt = heatmap(axs..., err_map',
    title="GP Variance",
    framestyle=:none,
    titlefontsize=26,
    colorbar_tickfontsize=18,
    # clim=(0, 1),
)
scatter!(x1[1:i], x2[1:i];
    label=false,
    color=:green,
    markersize=8)

weights = (1, 1e1, 1, 1)

sampleCost = DerivVar(
    occ, samples[1:i], bm, quantities, weights
)

new_loc = selectSampleLocation(sampleCost, occ.bounds)

data = -sampleCost.(points)

data[occ] .= NaN

der_plt = heatmap(axs..., data',
    title="Gradient Norm",
    framestyle=:none,
    titlefontsize=26,
    colorbar_tickfontsize=18,
    # clim=(0, 1),
)
scatter!(x1[1:i], x2[1:i];
    label=false,
    color=:green,
    markersize=8)
scatter!([new_loc[1]], [new_loc[2]],
    label=false,
    color=:red,
    shape=:xcross,
    markersize=14)

sampleCost = EIGF(
    occ, samples[1:i], bm, quantities, weights
)

new_loc = selectSampleLocation(sampleCost, occ.bounds)

data = -sampleCost.(points)

data[occ] .= NaN

eigf_plt = heatmap(axs..., data',
    title="Nearest-Sample Diff",
    framestyle=:none,
    titlefontsize=26,
    colorbar_tickfontsize=18,
    # clim=(0, 1),
)
scatter!(x1[1:i], x2[1:i];
    label=false,
    color=:green,
    markersize=8)
scatter!([new_loc[1]], [new_loc[2]],
    label=false,
    color=:red,
    shape=:xcross,
    markersize=14)

plt = plot(
    pred_plt, err_plt,
    der_plt, eigf_plt,
    layout = 4,
    size=(1000,800)
)
# display(plt)

savefig(output_dir * "thesis/obj_func_comp.png")

#* dist-scaled objective function plots

# eigf, dist-scaled

using .SampleCosts, .Samples, .Maps

pyplot()

# mean, var, derivvar, eigf

i = 20

bm = beliefs[i]

axs, points = generateAxes(occ)

weights = (1, 1e2, 1, 1)


sampleCost = EIGF(
    occ, samples[1:i], bm, quantities, weights
)

new_loc_eigf = selectSampleLocation(sampleCost, occ.bounds)

data_eigf = -sampleCost.(points)

data_eigf[occ] .= NaN

sampleCost = DistScaledEIGF(
    occ, samples[1:i], bm, quantities, weights
)

new_loc_dist = selectSampleLocation(sampleCost, occ.bounds)

data_dist = -sampleCost.(points)

data_dist[occ] .= NaN


eigf_plt = heatmap(axs..., data_eigf',
    title="No Distance Scaling",
    framestyle=:none,
    titlefontsize=24,
    colorbar_tickfontsize=18,
    # clim=(0, 1),
)
scatter!(x1[1:i-1], x2[1:i-1];
    label=false,
    color=:green,
    markersize=8)
scatter!(x1[i:i], x2[i:i];
    label=false,
    color=:royalblue,
    shape=:utriangle,
    markersize=14)
scatter!([new_loc_eigf[1]], [new_loc_eigf[2]],
    label=false,
    color=:red,
    shape=:xcross,
    markersize=14)


dist_plt = heatmap(axs..., data_dist',
    title="Distance Scaling",
    framestyle=:none,
    titlefontsize=24,
    colorbar_tickfontsize=18,
    # clim=(0, 1),
)
scatter!(x1[1:i-1], x2[1:i-1];
    label=false,
    color=:green,
    markersize=8)
scatter!(x1[i:i], x2[i:i];
    label=false,
    color=:royalblue,
    shape=:utriangle,
    markersize=14)
scatter!([new_loc_dist[1]], [new_loc_dist[2]],
    label=false,
    color=:red,
    shape=:xcross,
    markersize=14)


plt = plot(
    eigf_plt, dist_plt,
    layout=2,
    size=(1000, 400)
)
# display(plt)

savefig(output_dir * "thesis/dist_scaled_comp.png")

# eigf, dist-scaled

#* dist-scaled run sample order

pyplot()

dir = "new_syn/syn_multiKernel_means_noises_fullpdf_nodrop_EIGF"
file_name = output_dir * "$dir/data_000" * output_ext

data = load(file_name)
mission = data["missions"][1].mission
samples = data["missions"][1].samples
beliefs = data["missions"][1].beliefs

occ = mission.occupancy
quantities = eachindex(mission.sampler)
num_quant = length(mission.sampler)

xp = first.(getfield.(samples, :x))
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)

bm = beliefs[end]

axs, points = generateAxes(occ)
pred_map, err_map = bm(tuple.(points, 1))

# blocked points
pred_map[occ] .= NaN
err_map[occ] .= NaN

eigf_plt = heatmap(axs..., pred_map',
    title="Predictions",
    framestyle=:none,
    titlefontsize=26,
    colorbar_tickfontsize=18,
    clim=(0, 1),
)
scatter!(x1, x2;
    label=false,
    color=colormap("Greens", length(xp)),
    markersize=10)
plot!(x1, x2, label=false, color=:gray, line=:dash)

dxs = map(xp) do x
    dx1, dx2 = 0, 1
    if abs(x[1] - 0) < 0.05
        dx1 = 1
    end
    if abs(x[1] - 1) < 0.05
        dx1 = -1
    end
    if abs(x[2] - 0) < 0.05
        dx2 = 1
    end
    if abs(x[2] - 1) < 0.05
        dx2 = -1.25
    end
    n = 1
    dx1/20/n, dx2/20/n
end
dx1 = getindex.(dxs, 1)
dx2 = getindex.(dxs, 2)
for (i, (x1, x2)) in enumerate(xp)
    annotate!(x1 .+ dx1[i], x2 .+ dx2[i], Plots.text(string(i), :lightgray, 16))
end

dir = "new_syn/syn_multiKernel_means_noises_fullpdf_nodrop_DistScaledEIGF"
file_name = output_dir * "$dir/data_000" * output_ext

data = load(file_name)
mission = data["missions"][1].mission
samples = data["missions"][1].samples
beliefs = data["missions"][1].beliefs

occ = mission.occupancy
quantities = eachindex(mission.sampler)
num_quant = length(mission.sampler)

xp = first.(getfield.(samples, :x))
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)

bm = beliefs[end]

axs, points = generateAxes(occ)
pred_map, err_map = bm(tuple.(points, 1))

# blocked points
pred_map[occ] .= NaN
err_map[occ] .= NaN

palette(:greens, length(xp))

dist_plt = heatmap(axs..., pred_map',
    title="Predictions",
    framestyle=:none,
    titlefontsize=26,
    colorbar_tickfontsize=18,
    clim=(0, 1),
)
scatter!(x1, x2;
    label=false,
    color=colormap("Greens", length(xp)),
    markersize=10)
plot!(x1, x2, label=false, color=:gray, line=:dash)

dxs = map(xp) do x
    dx1, dx2 = 0, 1
    if abs(x[1] - 0) < 0.05
        dx1 = 1
    end
    if abs(x[1] - 1) < 0.05
        dx1 = -1
    end
    if abs(x[2] - 0) < 0.05
        dx2 = 1
    end
    if abs(x[2] - 1) < 0.05
        dx2 = -1.25
    end
    n = 1
    dx1/20/n, dx2/20/n
end
dx1 = getindex.(dxs, 1)
dx2 = getindex.(dxs, 2)
for (i, (x1, x2)) in enumerate(xp)
    annotate!(x1 .+ dx1[i], x2 .+ dx2[i], Plots.text(string(i), :lightgray, 16))
end

plt = plot(
    eigf_plt, dist_plt,
    layout=2,
    size=(1000, 400)
)
# display(plt)

savefig(output_dir * "thesis/dist_scaled_run_comp.png")

#* mto single runs

#** load

gr()

dir = "new_syn/syn_mtoKernel_means_noises_fullpdf_nodrop_DistScaledEIGF"
file_name = output_dir * "$dir/data_101" * output_ext
data = load(file_name)
missions = data["missions"]

#** run

plts = map(missions) do mission
    beliefs = mission.beliefs
    cors = map(beliefs) do bm
        outputCorMat(bm)[2:end, 1]
    end
    plot(stack(cors, dims=1), ylim=(-1, 1))
end

plot(plts..., layout=(3,6), size=(1200, 800)) |> display
