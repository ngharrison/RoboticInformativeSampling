
using Random: seed!
using LinearAlgebra: I
using Statistics: cor
using Plots

using InformativeSampling

using .Maps: Map, generateAxes
using .Samples: Sample, MapsSampler

using InformativeSamplingUtils
using .DataIO: output_dir, GaussGroundTruth, Peak

#* data

mission_peaks = [3,3,4,4,5,5]
priors = Bool[1,1,1];
bounds = (lower=[0.0,0.0], upper=[1.0,1.0])
occupancy = Map(zeros(Bool, 100, 100), bounds)
axs, points = generateAxes(occupancy)

data = map(enumerate(mission_peaks)) do (seed_val, num_peaks)
    seed!(seed_val) # make random values deterministic

    # simulated
    peaks = [Peak(rand(2).*(bounds.upper-bounds.lower) .+ bounds.lower, 0.02*(rand()+0.5)*I, rand())
            for i in 1:num_peaks]
    ggt = GaussGroundTruth(peaks)
    mat = ggt(points)
    Map(mat./maximum(mat), bounds)
end

#* qoi maps

cy = 0:.001:1
p = plot(
    # the plots
    (heatmap(axs..., d', ticks=false, framestyle=:none) for d in data)...,
    # and the colorbar
    heatmap([0], cy, reshape(cy, :, 1), xticks=false, yticks=0:0.5:1, mirror=true);
    layout=@layout([grid(2, 3){0.95w} c]),
    clim=(0, 1),
    colorbar=false,
    size=(1500, 800),
    plot_title="Synthetic Quantities",
    plot_titlefontsize=26,
    tickfontsize=20,
    # margin=4Plots.mm
)

# display(p)

savefig(output_dir * "paper/all_qois.png")
