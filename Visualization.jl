module Visualization

using Plots
using AbstractGPs: AbstractGP as AGP
using Initialization
using BeliefModel
using Sampling

export visualize

default_res = [0.01, 0.01]
sample_color = :green
new_sample_color = :red

# main function to visualize everything
function visualize(region, samples, gt::GT, belief_model::AGP, weights)
    l = @layout [a ; b c]
    plot(
        visualize(region, samples, belief_model),
        visualize(region, gt),
        visualize(region, samples, belief_model, weights),
        layout=l
    )
end

# functions to visualize individual pieces
function visualize(region, gt::GT; res=default_res)
    axis, points = getAxes(region; res)
    Y = gt(points)
    map = reshape(Y, length.(axis)...)
    heatmap(axis..., map;
            xlabel="x1",
            ylabel="x2",
            title="Ground Truth"
            )
end

# show side-by-side of belief_model mean and std
function visualize(region, samples, belief_model::AGP; res=default_res)
    axis, points = getAxes(region; res)
    m1, m2 = length.(axis)
    μ, σ = getBelief(points, belief_model)
    pred_map = reshape(μ, m1,m2)
    err_map = reshape(σ, m1,m2)

    xp = getfield.(samples, :x)
    x1 = getindex.(xp, 1)
    x2 = getindex.(xp, 2)

    p1 = heatmap(axis..., pred_map)
    scatter!(x1[1:end-1], x2[1:end-1];
             xlabel="x1",
             ylabel="x2",
             title="GP Mean",
             legend=nothing,
             color=sample_color
             )
    scatter!([x1[end]], [x2[end]], color=new_sample_color)

    p2 = heatmap(axis..., err_map)
    scatter!(x1[1:end-1], x2[1:end-1];
             xlabel="x1",
             ylabel="x2",
             title="GP Std",
             legend=nothing,
             color=sample_color
             )
    scatter!([x1[end]], [x2[end]], color=new_sample_color)

    plot(p1, p2)
end

# show cost function values
function visualize(region, samples, belief_model::AGP, weights; res=default_res)
    axis, points = getAxes(region; res)
    vals = -createCostFunc(region, samples, belief_model, weights).(points)
    map = reshape(vals, length.(axis)...)

    xp = getfield.(samples, :x)
    x1 = getindex.(xp, 1)
    x2 = getindex.(xp, 2)

    heatmap(axis..., map)
    scatter!(x1[1:end-1], x2[1:end-1];
             xlabel="x1",
             ylabel="x2",
             title="Obj Function",
             legend=nothing,
             color=sample_color
             )
    scatter!([x1[end]], [x2[end]], color=new_sample_color)
end

function getAxes(region; res=default_res)
    axis = [region.x1.lb:res[1]:region.x1.ub,
            region.x2.lb:res[2]:region.x2.ub]
    points = [[x1, x2] for x1 in axis[1] for x2 in axis[2]]
    return axis, points
end

# TODO: decide about using this code

# # placeholders to avoid recomputing
# axis = nothing
# points = nothing
# default_res = [0.01, 0.01] # default values
#
# function getAxes(region; res=nothing)
#     global axis
#     global points
#     global default_res
#     if res !== nothing && res !== default_res # recompute
#         default_res = res
#         axis, points = generateAxes(region; res)
#     elseif axis === nothing # recompute
#         global res
#         axis, points = generateAxes(region; res)
#     end
#     return axis, points
# end
#
# function generateAxes(region; res)
#     global axis = [region.x1.lb:res[1]:region.x1.ub,
#                    region.x2.lb:res[2]:region.x2.ub]
#     global points = [[i,j] for i in axis[1] for j in axis[2]]
#     return axis, points
# end

end
