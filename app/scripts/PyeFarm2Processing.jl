
using InformativeSampling

using InformativeSamplingUtils
using .DataIO, .Visualization

using FileIO, JLD2

#* load data

dir = "pye_farm_trial2/"
file_names = readdir(output_dir * dir * "dense")

samples = []
for file_name in file_names
    data = load(output_dir * dir * "dense/" * file_name)
    append!(samples, data["metrics"])
end

#* save

jldsave(output_dir * dir * file_names[end]; samples)


#* package mission data

using OrderedCollections: OrderedDict
using YAML

# maintain printed form of arrays in yaml
struct ArrayWrapper{T<:AbstractArray}
    a::T
end
Base.show(io::IO, a::ArrayWrapper{T}) where T = print(io, "$(a.a)")

dir = "pye_farm_trial2/"
file_names = filter(n->!any(contains.(n, ("strange", "grid"))), readdir(output_dir * dir * "named"))

for file_name in file_names
    data = load(output_dir * dir * "named/" * file_name)
    m = data["mission"]

    n = OrderedDict(key => getfield(m, key) for key ∈ fieldnames(typeof(m)))
    n[:start_locs] = ArrayWrapper(n[:start_locs])

    base_name = splitext(file_name)[1]
    mkpath(output_dir * dir * "packaged/$(base_name)/")
    YAML.write_file(output_dir * dir * "packaged/$(base_name)/"
                    * "mission.yml", n)
end

#* save belief model maps and samples

using DelimitedFiles

using .BeliefModels: BeliefModel, outputCorMat
using .Maps

dir = "pye_farm_trial2/"
file_names = filter(n->!any(contains.(n, ("strange", "grid"))),
                    readdir(output_dir * dir * "named"))

for file_name in file_names
    base_name = splitext(file_name)[1]

    data = load(output_dir * dir * "named/" * file_name)
    mission = data["mission"]
    samples = data["samples"]
    beliefs = data["beliefs"]

    new_samples = (s->((s.x...,), s.y)).(samples)
    writedlm(output_dir * dir * "packaged/" * base_name * "/samples.txt", [new_samples], "\n")

    cors = [outputCorMat(bm)[:, 1] for bm in beliefs]

    writedlm(output_dir * dir * "packaged/" * base_name * "/correlations.txt", [cors], "\n")

    belief_maps = produceMaps(beliefs[end], mission.occupancy)
    m, s = mapToImg.(belief_maps)

    writedlm(output_dir * dir * "packaged/" * base_name * "/avg_height_belief.csv", m, ',')
    writedlm(output_dir * dir * "packaged/" * base_name * "/avg_height_uncertainty.csv", s, ',')

    # currently exporting maps to only 25x25 for comparison with dense sampling
    mkpath(output_dir * dir * "packaged/" * base_name * "/avg_height_beliefs_25x25")
    mkpath(output_dir * dir * "packaged/" * base_name * "/avg_height_uncertainties_25x25")
    mkpath(output_dir * dir * "packaged/" * base_name * "/sample_utilities_25x25")

    height_samples = filter(s->s.x[2]==1, samples)

    for (i, belief) in enumerate(beliefs)
        belief_maps = produceMaps(belief, mission.occupancy.bounds, (25, 25))
        m, s = mapToImg.(belief_maps)

        sampleCost = mission.sampleCostType(mission.occupancy, height_samples[1:i], belief, 1:1, mission.weights)
        axs, points = generateAxes(mission.occupancy.bounds, (25, 25))
        u = mapToImg(-sampleCost.(points))

        n = lpad(i, 2, '0')
        writedlm(output_dir * dir * "packaged/" * base_name * "/avg_height_beliefs_25x25/$n.csv", m, ',')
        writedlm(output_dir * dir * "packaged/" * base_name * "/avg_height_uncertainties_25x25/$n.csv", s, ',')
        writedlm(output_dir * dir * "packaged/" * base_name * "/sample_utilities_25x25/$n.csv", u, ',')
    end

    ndvi_samples = filter(s->s.x[2]==2, samples)
    if !isempty(ndvi_samples)
        bm = BeliefModel(ndvi_samples, mission.occupancy.bounds)

        belief_maps = produceMaps(bm, mission.occupancy.bounds, (25, 25); quantity=2)
        m, s = mapToImg.(belief_maps)

        writedlm(output_dir * dir * "packaged/" * base_name * "/avg_ndvi_belief.csv", m, ',')
        writedlm(output_dir * dir * "packaged/" * base_name * "/avg_ndvi_uncertainty.csv", s, ',')
    end
end

#* save dense grid maps

dir = "pye_farm_trial2/"

file_name = output_dir * dir * "named/50x50.jld2"
data = load(file_name)
mission = data["mission"]

file_name = output_dir * dir * "named/50x50_dense_grid.jld2"
data = load(file_name)
samples = data["samples"]

base_name = output_dir * dir * "packaged/50x50_dense_grid"
mkpath(base_name)

new_samples = (s->((s.x...,), s.y)).(samples)
writedlm(base_name * "/samples.txt", [new_samples], "\n")

height_samples = filter(s->s.x[2]==1, samples)
ndvi_samples = filter(s->s.x[2]==2, samples)

using .Maps

d = Int(sqrt(length(height_samples)))
m_grid = Map(zeros(d,d), mission.bounds)
for s in height_samples
    m_grid[pointToCell(s.x[1], m_grid)] = s.y
end
writedlm(base_name * "/avg_height.csv", mapToImg(m_grid), ',')

d = Int(sqrt(length(height_samples)))
m_grid = Map(zeros(d,d), mission.bounds)
for s in ndvi_samples
    m_grid[pointToCell(s.x[1], m_grid)] = s.y
end
writedlm(base_name * "/avg_ndvi.csv", mapToImg(m_grid), ',')

bm = BeliefModel(height_samples, mission.occupancy.bounds; mission.noise)
belief_maps = produceMaps(bm, mission.occupancy)
m, s = mapToImg.(belief_maps)

writedlm(base_name * "/avg_height_belief.csv", m, ',')
writedlm(base_name * "/avg_height_uncertainty.csv", s, ',')

bm = BeliefModel(ndvi_samples, mission.occupancy.bounds; mission.noise)
belief_maps = produceMaps(bm, mission.occupancy; quantity=2)
m, s = mapToImg.(belief_maps)

writedlm(base_name * "/avg_ndvi_belief.csv", m, ',')
writedlm(base_name * "/avg_ndvi_uncertainty.csv", s, ',')

#* look at correlation between height and ndvi

using Statistics

vals = map((height_samples, ndvi_samples)) do a
    getfield.(sort(a, by=s->s.x[1]), :y)
end

cor(vals...)

#*

pyplot()

#* some maps

using .BeliefModels
using .Maps

using Plots


dir = "pye_farm_trial2/named/"

file_name = output_dir * dir * "50x50.jld2"
data = load(file_name)
mission = data["mission"]
bounds = getBounds(mission.occupancy)

axs, points = generateAxes(mission.occupancy)

ranges = [(Inf, -Inf), (Inf, -Inf)]
names = ["50x50", "50x50_dense_grid"]

# will be (sampling, quantity)
pred_maps = map(names) do name
    file_name = output_dir * dir * "$name.jld2"
    data = load(file_name)
    samples = data["samples"]

    things = [("height", "Estimated Plant Heights (mm)"),
              ("ndvi", "Estimated NDVI")]

    map(enumerate(things)) do (i, (quantity, title))
        q_samples = filter(s -> s.x[2] == i, samples)

        bm = BeliefModel(q_samples, bounds; mission.noise)
        beliefs = bm(tuple.(vec(points), i))
        pred_map, err_map = reshape.(beliefs, Ref(size(mission.occupancy)))

        if quantity == "ndvi"
            pred_map ./= 255
        end

        global ranges[i] = (min(minimum(pred_map), ranges[i][1]), max(maximum(pred_map), ranges[i][2]))

        # xp = first.(getfield.(q_samples, :x))
        # x1 = getindex.(xp, 1)
        # x2 = getindex.(xp, 2)

        heatmap(axs..., pred_map';
            title,
            framestyle=:none,
            ticks=false,
            size=(600, 500),
            titlefontsize=21,
            colorbar_tickfontsize=20,
            legendfontsize=14,
            aspect_ratio=:equal,
            right_margin=7Plots.mm
        )
        # scatter!(x1, x2;
        #     label=false,
        #     color=:green,
        #     legend=(0.15, 0.87),
        #     markersize=8)
        # gui()

        savefig(output_dir * "jfr_paper/$(name)_$(quantity).png")

        return pred_map
    end
end


#* another

p1 = heatmap(axs..., pred_maps[1][1]';
    title="Estimated Height (mm)",
    clim=ranges[1]
)
# scatter!(x1, x2;
#          label=false,
#          color=:green,
#          markersize=8)

p2 = heatmap(axs..., pred_maps[2][1]';
    title="Estimated Height (mm)",
    clim=ranges[1]
)
# scatter!(x1, x2;
#          label=false,
#          color=:green,
#          markersize=8)

p = plot(p1, p2;
    layout=(1, 2),
    framestyle=:none,
    ticks=false,
    size=(1000, 400),
    titlefontsize=21,
    colorbar_tickfontsize=20,
    legendfontsize=14,
    aspect_ratio=:equal,
    right_margin=4Plots.mm
    # left_margin=4Plots.mm,
)
gui()

savefig(output_dir * "jfr_paper/sampling_comparison.png")

#* all four in a grid

# sampling
# quantity

plots = fill(plot(), 2, 2)

i, j = 1, 1
plots[i,j] = heatmap(pred_maps[i][j]';
    title="Sparse Sampling",
    ylabel="Plant Height (mm)",
    clim=ranges[j]
)
i, j = 2, 1
plots[i,j] = heatmap(pred_maps[i][j]';
    title="Dense Sampling",
    clim=ranges[j]
)
i, j = 1, 2
plots[i,j] = heatmap(pred_maps[i][j]';
    ylabel="NDVI",
    seriescolor=:YlGn,
    clim=ranges[j]
)
i, j = 2, 2
plots[i,j] = heatmap(pred_maps[i][j]';
    seriescolor=:YlGn,
    clim=ranges[j]
)

p = plot(plots...;
    framestyle=:none,
    ticks=false,
    size=(850, 700),
    titlefontsize=25,
    labelfontsize=21,
    colorbar_tickfontsize=20,
    legendfontsize=14,
    aspect_ratio=:equal,
)

savefig(output_dir * "jfr_paper/50x50_results_grid.png")


#* data

mission, = simMission(; seed_val=3, num_peaks=4, priors=collect(Bool, (0,0,0)))

@time samples, beliefs = mission(sleep_time=0.0);

j = 12
axs, points = generateAxes(mission.occupancy)
dims = Tuple(length.(axs))
μ, σ = beliefs[j](tuple.(vec(points), 1))
pred_map = reshape(μ, dims)
err_map = reshape(σ, dims)
xp = first.(getfield.(samples[1:j], :x))
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)

#* plot maps

p1 = heatmap(axs..., pred_map',
             title="Estimated Values",
             ticks=false,
             )
scatter!(x1, x2;
         framestyle=:none,
         aspect_ratio=:equal,
         label="Samples",
         color=:green,
         legend=(0.02, 0.75),
         markersize=10)
p2 = heatmap(axs..., err_map',
             title="Uncertainties",
             ticks=false,
             )
scatter!(x1, x2;
         framestyle=:none,
         aspect_ratio=:equal,
         label="Samples",
         color=:green,
         legend=(0.02, 0.75),
         markersize=10)

plot(p1, p2,
     size=(1400, 600),
     titlefontsize=40,
     tickfontsize=20,
     legendfontsize=22,
     colorbar_tickfontsize=30,
     )

savefig(output_dir * "jfr_paper/example_gp_pyplot.png")
