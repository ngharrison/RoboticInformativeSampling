# script for analyzing data from missions
# can be run after Main.jl or by opening a saved file

using Plots

f = jldopen("../output/2023-07-21-17-56-40.jdl2")

mission = f["mission"]
beliefs = f["beliefs"]
samples = f["samples"]
metrics = f["metrics"]

plot(
    plot(1:mission.num_samples, metrics.mae),
    plot(1:mission.num_samples, metrics.mu),
    plot(1:mission.num_samples, metrics.mb),
    layout = (3, 1)
)
