using Plots
using Plots: mm

using AdaptiveSampling: Maps, Missions

using .Maps: res, generateAxes
using .Missions: simMission

function copy_ticks(plt::Plots.Plot=current())
    sp::Plots.Subplot = plt[1]
    ptx = twinx(sp)
    plot!(ptx,xlims=xlims(plt),ylims=ylims(plt),xformatter=_->"",yformatter=_->"")
    pty = twiny(sp)
    plot!(pty,xlims=xlims(plt),ylims=ylims(plt),xformatter=_->"",yformatter=_->"")
end

mission = simMission(; seed_val=3, num_peaks=4, priors=collect(Bool, (0,0,0)))

@time samples, beliefs = mission(visuals=false, sleep_time=0.0);

j = 12
axs, points = generateAxes(mission.occupancy)
dims = Tuple(length.(axs))
μ, σ = beliefs[j](tuple.(vec(points), 1))
pred_map = reshape(μ, dims)
err_map = reshape(σ, dims)
xp = first.(getfield.(samples[1:j], :x))
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)
## plot maps
p1 = heatmap(axs..., pred_map', title="Predicted Values", ticks=false)
# copy_ticks()
scatter!(x1, x2;
         framestyle=:none,
         aspect_ratio=:equal,
         label="Samples",
         color=:green,
         legend=(0.2, 0.87),
         markersize=6)
l1, h1 = extrema(pred_map)
cy1 = l1:.001*(h1-l1):h1
cbar1 = heatmap([0], cy1, reshape(cy1, :, 1),
               # aspect_ratio=1/100,
               xticks=false,
               yticks=false,
               # yticks=l1:0.25*(h1-l1):h1,
               mirror=true
               )
p2 = heatmap(axs..., err_map', title="Uncertainties", ticks=false)
# copy_ticks()
scatter!(x1, x2;
         framestyle=:none,
         aspect_ratio=:equal,
         label="Samples",
         color=:green,
         legend=(0.2, 0.87),
         markersize=6)
l2, h2 = extrema(err_map)
cy2 = l2:.001*(h2-l2):h2
cbar2 = heatmap([0], cy2, reshape(cy2, :, 1),
               # aspect_ratio=1/100,
               xticks=false,
               yticks=false,
               # yticks=l2:0.25*(h2-l2):h2,
               mirror=true
               )

plot(p1, p2,
     size=(1400, 600),
     titlefontsize=30,
     tickfontsize=20,
     legendfontsize=18,
     right_margin=17mm,
     )

savefig(output_dir * "paper/example_gp.png")
