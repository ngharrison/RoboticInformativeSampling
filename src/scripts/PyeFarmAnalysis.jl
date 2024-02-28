# allows using modules defined in any file in project src directory
mod_dir = dirname(Base.active_project()) * "/src/modules"
if mod_dir ∉ LOAD_PATH
    push!(LOAD_PATH, mod_dir)
end

using Maps, Missions, BeliefModels, Samples, Outputs, ROSInterface, Visualization, Outputs

using Statistics, FileIO, Plots, Images
using Plots: mm

# pyplot()

file_name = output_dir * "2024-02-15-12-02-46_mission.jld2"
file_name = output_dir * "2024-02-15-12-22-16_mission.jld2"
file_name = output_dir * "2024-02-15-12-56-06_mission.jld2"

data = load(file_name)
mission = data["mission"]
beliefs = data["beliefs"]
samples = data["samples"]
quantities = eachindex(mission.sampler)
sampleCost = mission.sampleCostType(mission, samples, beliefs[end], quantities)
vis(mission, samples, beliefs[end], mission.occupancy.lb; quantity=1)


## concatenate split missions
file_name1 = output_dir * "pye_farm_trial/2024-02-15-17-26-06_mission.jld2"
file_name2 = output_dir * "pye_farm_trial/2024-02-15-18-26-15_mission.jld2"

data1 = load(file_name1)
data2 = load(file_name2)
samples = [data1["samples"]; data2["samples"]]
mission = data1["mission"]
lb, ub = mission.occupancy.lb, mission.occupancy.ub
beliefs = map(1:length(samples)) do i
    BeliefModel(samples[1:i], lb, ub)
end
save(mission, samples, beliefs; sub_dir_name="pye_farm_trial_gen")

## elevation maps

elev_img = load(maps_dir * "dem_50x50.tif")
elevMap = imgToMap(gray.(elev_img), lb, ub)
vis(elevMap)

elev_img = load(maps_dir * "dem_15x15.tif")
elevMap = imgToMap(gray.(elev_img), lb, ub)
vis(elevMap)

## load mission data
name = "pye_farm_trial_named/30samples_50x50_priors"
file_name = output_dir * "$(name)" * output_ext
data = load(file_name)
mission = data["mission"]
samples = data["samples"]
beliefs = data["beliefs"]
lb, ub = mission.occupancy.lb, mission.occupancy.ub

using Logging
global_logger(ConsoleLogger(stderr, Debug))

## replay mission
replay(mission, samples, beliefs; sleep_time=1.0)

## make png from proxy ground truth
saveBeliefMapToPng(beliefs[end], mission.occupancy, name)

## look at every sample collected together
all_samples = map(readdir(output_dir * "pye_farm_trial", join=true)[2:end]) do file_name
    data = load(file_name)
    data["samples"]
end |> Iterators.flatten |> collect

x = first.(first.(getfield.(all_samples, :x)))
y = last.(first.(getfield.(all_samples, :x)))
z = getfield.(all_samples, :y)

display(scatter3d(x, y, z, legend=false))

mission = load(readdir(output_dir * "pye_farm_trial", join=true)[end-1])["mission"]
lb, ub = mission.occupancy.lb, mission.occupancy.ub
beliefModel = BeliefModel(all_samples, lb, ub)

vis(beliefModel, [], mission.occupancy)


## with different prior samples

name = "pye_farm_trial_named/30samples_50x50_priors"
file_name = output_dir * "$(name)" * output_ext
data = load(file_name)
mission = data["mission"]
samples = data["samples"]
beliefs = data["beliefs"]
lb, ub = mission.occupancy.lb, mission.occupancy.ub

elev_img = load(maps_dir * "dem_15x15.tif")
elevMap = imgToMap(gray.(elev_img), lb, ub)
prior_maps = [elevMap]

n = (7,7)
axs_sp = range.(lb, ub, n)
points_sp = vec(collect.(Iterators.product(axs_sp...)))
prior_samples = [Sample{Float64}((x, i+length(mission.sampler)), d(x))
                 for (i, d) in enumerate(prior_maps)
                     for x in points_sp if !isnan(d(x))]

beliefs = map(1:length(samples)) do i
    BeliefModel([prior_samples; samples[1:i]], lb, ub)
end

vis(elevMap; points=points_sp)

##
cors = map(beliefs) do b
    outputCorMat(b)
end

using LinearAlgebra

i = 10
outputCorMat(beliefs[i])
for i in 1:30
    vis(mission, samples[1:i], beliefs[i], nothing)
    sleep(1.5)
end

prior_belief = BeliefModel(prior_samples, lb, ub)
vis(mission, prior_samples, prior_belief, nothing; quantity=2)


## combined points

names = ["30samples_15x15_1", "30samples_15x15_2", "30samples_15x15_priors",
         "30samples_50x50", "30samples_50x50_priors", "100samples_50x50_grid"]

all_samples = map(names) do name
    file_name = output_dir * "pye_farm_trial_named/" * name * output_ext
    data = load(file_name)
    data["samples"]
end |> Iterators.flatten |> collect

x = first.(first.(getfield.(all_samples, :x)))
y = last.(first.(getfield.(all_samples, :x)))
z = getfield.(all_samples, :y)

display(scatter(x, y, legend=false))

display(scatter3d(x, y, z, legend=false))

data = load(output_dir * "pye_farm_trial_named/" * names[end] * output_ext)
mission = data["mission"]
samples = data["samples"]
beliefs = data["beliefs"]
lb, ub = mission.occupancy.lb, mission.occupancy.ub

beliefs[end]

beliefModel = BeliefModel(all_samples, lb, ub)

vis(beliefs[end], [], mission.occupancy)

beliefModel = BeliefModel([prior_samples; samples], lb, ub)

outputCorMat(beliefModel)

##

gt_name = "100samples_50x50_grid"
gt_data = load(output_dir * "pye_farm_trial_named/" * gt_name * output_ext)
gt_belief = gt_data["beliefs"][end]
mission = gt_data["mission"]
occupancy = mission.occupancy
lb, ub = occupancy.lb, occupancy.ub

axes, points = generateAxes(occupancy)
dims = Tuple(length.(axes))
μ, σ = gt_belief(tuple.(vec(points), 1))
gt_pred_map = Map(reshape(μ, dims), lb, ub)

name = "30samples_50x50_priors"
file_name = output_dir * "pye_farm_trial_named/" * name * output_ext
data = load(file_name)
beliefs = data["beliefs"]

for i in 1:30
    μ, σ = beliefs[i](tuple.(vec(points), 1))
    pred_map = Map(reshape(μ, dims), lb, ub)
    err_map = abs.(pred_map .- gt_pred_map)
    vis(gt_pred_map, pred_map; titles=["gt", "belief"])
    println(mean(err_map))
    sleep(1.0)
end

maes = map(beliefs) do belief
    μ, σ = belief(tuple.(vec(points), 1))
    pred_map = reshape(μ, dims)
    mean(abs.(pred_map .- gt_pred_map))
end

display(scatter(maes))

## create slides

names = ["30samples_15x15_1", "30samples_15x15_2", "30samples_15x15_priors",
         "30samples_50x50", "30samples_50x50_priors", "100samples_50x50_grid"]

all_samples = map(names) do name
    name = names[1]
    file_name = output_dir * "pye_farm_trial_named/" * name * output_ext
    data = load(file_name)
    beliefs = data["beliefs"]
end |> Iterators.flatten |> collect

## other plots
using Statistics, FileIO, GLMakie, Images

x = range(0, 10, length=100)
y = sin.(x)
lines(x, y)
lines(sin)

name = "pye_farm_trial_named/30samples_50x50_priors"
file_name = output_dir * "$(name)" * output_ext
data = load(file_name)
mission = data["mission"]
samples = data["samples"]
beliefs = data["beliefs"]
occupancy = mission.occupancy
lb, ub = occupancy.lb, occupancy.ub
axes, points = generateAxes(occupancy)
dims = Tuple(length.(axes))
μ, σ = beliefs[15](tuple.(vec(points), 1))
pred_map = Map(reshape(μ, dims), lb, ub)

fig, ax, hm = heatmap(pred_map')
Colorbar(fig[1, 2], hm, width=20)
scatter!([20], [40], color=:black, markersize=10)
display(fig)

using GLMakie
f = Figure()
ax = GLMakie.Axis(f[1,1])

heatmap.([pred_map, pred_map])

## combined satellite and elevation map
lb, ub = [0.0, 0.0], [50.0, 50.0]
elev_img = Float64.(gray.(load(maps_dir * "dem_50x50.tif")))
elevMap = imgToMap((elev_img.-minimum(elev_img)).*100, lb, ub)

sat_img = load(maps_dir * "satellite_50x50.tif")

# sample sparsely from the prior maps
# currently all data have the same sample numbers and locations
n = (7,7) # number of samples in each dimension
axs_sp = range.(lb, ub, n)
points_sp = vec(collect.(Iterators.product(axs_sp...)))

p1 = plot(sat_img;
          title="Satellite Image")
p2 = heatmap(range.(lb, ub, size(elevMap))..., elevMap;
             title="Elevation Change (cm)",
             colorbar_tickfontsize=18,
             # aspect_ratio=:equal,
             # left_margin=10mm,
             c=:oslo
             )
# scatter!(getindex.(points_sp, 1), getindex.(points_sp, 2);
#          label=false,
#          color=:green,
#          markersize=8)
p = plot(p1, p2;
     size=(750, 300),
     titlefontsize=18,
     framestyle=:none,
         margin_top=
     )
savefig(expanduser("~/Projects/sampling_system_paper/figures/sat_elev_50x50.png"))
display(p)

## test correlations
axs, points = generateAxes(elevMap)
ss = vec(Sample.(tuple.(points, 1), elevMap))
bm = BeliefModel(ss, lb, ub)
vis(bm, [], Map(zeros(Bool,size(elev_img)), lb, ub))

name = "pye_farm_trial_named/100samples_50x50_grid"
file_name = output_dir * "$(name)" * output_ext
data = load(file_name)
samples = data["samples"]
ss = vec(Sample.(tuple.(points, 2), elevMap))
bm = BeliefModel([ss; samples], lb, ub)
outputCorMat(bm)
vis(bm, [], Map(zeros(Bool,size(elev_img)), lb, ub); quantity=1)

# still not getting consistent or expected results
