# script for analyzing data from missions
# can be run after Main.jl or by opening a saved file

# allows using modules defined in any file in project src directory
src_dir = dirname(Base.active_project()) * "/src"
if src_dir ∉ LOAD_PATH
    push!(LOAD_PATH, src_dir)
end

using Missions: Mission
using BeliefModels: BeliefModel
using Samples: Sample
using Outputs: output_dir, output_ext

using Statistics: mean, std
using FileIO: load
using Plots

"""
Calculates the standard deviation across elements of an array when the elements
are arrays of arrays.
"""
function stdM(cors)
    m = mean(cors)
    d = cors .- Ref(mean(cors))
    l = [[v.^2 for v in u] for u in d]
    s = sum(l) / (length(cors) - 1)
    return [[sqrt(v) for v in u] for u in s]
end

p_acc = []
p_cor = []

# SOGP
priors = (0,0,0)
file_name = output_dir * "paper/metrics_$(join(priors))" * output_ext

data = load(file_name)
maes = [run.mae for run in data["metrics"]]
cors = [run.cors for run in data["metrics"]]
dets = [[v.^2 for v in u] for u in cors]

p = plot(mean(maes), yerr=std(maes))
push!(p_acc, p)
p = plot(hcat((c[2:end] for c in mean(dets))...)',
         yerr=hcat((c[2:end] for c in stdM(dets))...)')
push!(p_cor, p)

# Single correlations
for priors in ((1,0,0), (0,1,0), (0,0,1))
    file_name = output_dir * "paper/metrics_$(join(priors))" * output_ext

    data = load(file_name)
    maes = [run.mae for run in data["metrics"]]
    cors = [run.cors for run in data["metrics"]]
    dets = [[v.^2 for v in u] for u in cors]

    p = plot(mean(maes), yerr=std(maes))
    push!(p_acc, p)
    p = plot(hcat((c[2:end] for c in mean(dets))...)',
             yerr=hcat((c[2:end] for c in stdM(dets))...)')
    push!(p_cor, p)
end

# Double correlations
for priors in ((1,1,0), (1,0,1), (0,1,1))
    file_name = output_dir * "paper/metrics_$(join(priors))" * output_ext

    data = load(file_name)
    maes = [run.mae for run in data["metrics"]]
    cors = [run.cors for run in data["metrics"]]
    dets = [[v.^2 for v in u] for u in cors]

    p = plot(mean(maes), yerr=std(maes))
    push!(p_acc, p)
    p = plot(hcat((c[2:end] for c in mean(dets))...)',
             yerr=hcat((c[2:end] for c in stdM(dets))...)')
    push!(p_cor, p)
end

# Triple correlation
priors = (1,1,1)
file_name = output_dir * "paper/metrics_$(join(priors))" * output_ext

data = load(file_name)
maes = [run.mae for run in data["metrics"]]
cors = [run.cors for run in data["metrics"]]
dets = [[v.^2 for v in u] for u in cors]

p = plot(mean(maes), yerr=std(maes))
push!(p_acc, p)
p = plot(hcat((c[2:end] for c in mean(dets))...)',
         yerr=hcat((c[2:end] for c in stdM(dets))...)')
push!(p_cor, p)


plot(
    p_acc...,
    ylim=(0,0.25),
    layout=grid(4,2)
)

plot(
    p_cor...,
    layout=grid(4,2)
)
