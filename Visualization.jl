module Visualization

using Plots
using AbstractGPs: AbstractGP as AGP
using Environment
using BeliefModel
using Sampling

export visualize

# placeholders to avoid recomputing
axis = nothing
points = nothing

default_res = [0.01, 0.01]
sample_color = :green
new_sample_color = :red

# main function to visualize everything
function visualize(gt::GT, belief_model::AGP, samples, weights, region; res=default_res)
    l = @layout [a ; b c]
    plot(
        visualize(belief_model, samples, region; res),
        visualize(gt, region; res),
        visualize(belief_model, samples, weights, region; res),
        layout=l
    )
end

# functions to visualize individual pieces
function visualize(gt::GT, region; res=default_res)
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
function visualize(belief_model::AGP, samples, region; res=default_res)
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
function visualize(belief_model::AGP, samples, weights, region; res=default_res)
    axis, points = getAxes(region; res)
    vals = -createCostFunc(region, samples, belief_model, getBelief, weights).(points)
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

function getAxes(region; res=nothing)
    global axis
    global points
    global default_res
    if res !== nothing && res !== default_res # recompute
        default_res = res
        axis, points = generateAxes(region; res=default_res)
    elseif axis === nothing # recompute
        axis, points = generateAxes(region; res=default_res)
    end
    return axis, points
end

function generateAxes(region; res=default_res)
    global axis = [region.lb[i]:res[i]:region.ub[i] for i in 1:2]
    global points = [[x1,x2] for x1 in axis[1] for x2 in axis[2]]
    return axis, points
end

end
