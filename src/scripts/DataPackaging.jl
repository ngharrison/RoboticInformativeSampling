
# allows using modules defined in any file in project src directory
mod_dir = dirname(Base.active_project()) * "/src/modules"
if mod_dir ∉ LOAD_PATH
    push!(LOAD_PATH, mod_dir)
end

using Missions: Mission
using BeliefModels: BeliefModel, outputCorMat
using Samples: Sample
using SampleCosts: EIGFSampleCost
using ROSInterface

using Maps


function Maps.generateAxes(lb, ub, dims)
    axs = range.(lb, ub, dims)
    points = collect.(Iterators.product(axs...))
    return axs, points
end

function produceMaps(beliefModel::BeliefModel, lb, ub, dims; quantity=1)
    axs, points = generateAxes(lb, ub, dims)
    μ, σ = beliefModel(tuple.(vec(points), quantity))
    pred_map, err_map = (Map(reshape(v, dims), lb, ub) for v in (μ, σ))

    return pred_map, err_map
end

function produceMaps(beliefModel::BeliefModel, map::Map; quantity=1)
    axs, points = generateAxes(map)
    μ, σ = beliefModel(tuple.(vec(points), quantity))
    pred_map, err_map = (Map(reshape(v, size(map)), map.lb, map.ub) for v in (μ, σ))

    return pred_map, err_map
end

mapToImg(map) = reverse(permutedims(map, (2,1)), dims=1)


using FileIO, JLD2, DelimitedFiles

output_dir = dirname(Base.active_project()) * "/app/output/"

#* package mission data

using OrderedCollections: OrderedDict
using YAML

# maintain printed form of arrays in yaml
struct ArrayWrapper{T<:AbstractArray}
    a::T
end
Base.show(io::IO, a::ArrayWrapper{T}) where T = print(io, "$(a.a)")

dir = "pye_farm_trial_named/"
file_names = filter(n->(contains(n, ".jld2") && !contains(n, "grid")), readdir(output_dir * dir))

for file_name in file_names
    data = load(output_dir * dir * file_name)
    m = data["mission"]

    n = OrderedDict(key => getfield(m, key) for key ∈ fieldnames(typeof(m)))
    n[:start_loc] = ArrayWrapper(n[:start_loc])

    base_name = splitext(file_name)[1]
    mkpath(output_dir * dir * "packaged/$(base_name)/")
    YAML.write_file(output_dir * dir * "packaged/$(base_name)/"
                    * "mission.yml", n)
end

#* save belief model maps and samples

dir = "pye_farm_trial_named/"
file_names = filter(n->(contains(n, ".jld2") && !contains(n, "grid")), readdir(output_dir * dir))

for file_name in file_names
    base_name = splitext(file_name)[1]

    data = load(output_dir * dir * file_name)
    mission = data["mission"]
    samples = data["samples"]
    beliefs = map(enumerate(samples)) do (i, sample)
        BeliefModel([samples[1:i]; mission.prior_samples], mission.occupancy.lb, mission.occupancy.ub; σn=0.0)
    end

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

    for (i, belief) in enumerate(beliefs)
        belief_maps = produceMaps(belief, mission.occupancy.lb, mission.occupancy.ub, (25, 25))
        m, s = mapToImg.(belief_maps)

        sampleCost = mission.sampleCostType(mission, samples[1:i], belief, 1:1)
        axs, points = generateAxes(mission.occupancy.lb, mission.occupancy.ub, (25, 25))
        u = mapToImg(-sampleCost.(points))

        n = lpad(i, 2, '0')
        writedlm(output_dir * dir * "packaged/" * base_name * "/avg_height_beliefs_25x25/$n.csv", m, ',')
        writedlm(output_dir * dir * "packaged/" * base_name * "/avg_height_uncertainties_25x25/$n.csv", s, ',')
        writedlm(output_dir * dir * "packaged/" * base_name * "/sample_utilities_25x25/$n.csv", u, ',')
    end

end

#* save dense grid maps

dir = "pye_farm_trial_named/"
file_name = output_dir * dir * "30samples_50x50.jld2"
data = load(file_name)
mission = data["mission"]

file_name = output_dir * dir * "100samples_50x50_grid.jld2"
data = load(file_name)
samples = data["samples"]

base_name = output_dir * dir * "packaged/100samples_50x50_grid"
mkpath(base_name)

new_samples = (s->((s.x...,), s.y)).(samples)
writedlm(base_name * "/samples.txt", [new_samples], "\n")

height_samples = filter(s->s.x[2]==1, samples)

using Maps

d = Int(sqrt(length(height_samples)))
m_grid = Map(zeros(d,d), mission.occupancy.lb, mission.occupancy.ub)
for s in height_samples
    m_grid[pointToCell(s.x[1], m_grid)] = s.y
end
writedlm(base_name * "/avg_height.csv", mapToImg(m_grid), ',')

bm = BeliefModel(height_samples, mission.occupancy.lb, mission.occupancy.ub)
belief_maps = produceMaps(bm, mission.occupancy)
m, s = mapToImg.(belief_maps)

writedlm(base_name * "/avg_height_belief.csv", m, ',')
writedlm(base_name * "/avg_height_uncertainty.csv", s, ',')
