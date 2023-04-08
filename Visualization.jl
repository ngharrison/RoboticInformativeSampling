module Visualization

using Plots
using Environment
using BeliefModels
using Exploration

export visualize

# placeholders to avoid recomputing
axes = nothing
points = nothing

default_res = [0.01, 0.01]
sample_color = :green
new_sample_color = :red

# main function to visualize everything
function visualize(beliefModel::BeliefModel, gtMap::Map, samples, region; res=default_res)
    l = @layout [a ; b c]
    plot(
        visualize(beliefModel, samples, region; res),
        visualize(gtMap, region),
        visualize(region.obsMap, region; title="Obstacle Map"),
        layout=l
    )
end

## functions to visualize individual pieces

# show any map data
function visualize(map::Map, region; title="Map")
    axes = (:).(region.lb, map.res, region.ub)
    data = map()
    heatmap(axes..., data';
            xlabel="x1",
            ylabel="x2",
            title
            )
end

# more generic fallback
function visualize(groundTruth::GroundTruth, region; res=default_res)
    axes, points = getAxes(region; res)
    data = groundTruth(points)
    heatmap(axes..., data';
            xlabel="x1",
            ylabel="x2",
            title="Ground Truth"
            )
end

# show side-by-side of beliefModel mean and std
function visualize(beliefModel::BeliefModel, samples, region; res=default_res)
    axes, points = getAxes(region; res)
    dims = Tuple(length.(axes))
    μ, σ = beliefModel(vec(points))
    pred_map = reshape(μ, dims)
    err_map = reshape(σ, dims)

    xp = getfield.(samples, :x)
    x1 = getindex.(xp, 1)
    x2 = getindex.(xp, 2)

    p1 = heatmap(axes..., pred_map')
    scatter!(x1[1:end-1], x2[1:end-1];
             xlabel="x1",
             ylabel="x2",
             title="GP Mean",
             legend=nothing,
             color=sample_color
             )
    scatter!([x1[end]], [x2[end]], color=new_sample_color)

    p2 = heatmap(axes..., err_map')
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
function visualize(costFunction::CostFunction, samples, region; res=default_res)
    axes, points = getAxes(region; res)
    data = -costFunction.(points)

    xp = getfield.(samples, :x)
    x1 = getindex.(xp, 1)
    x2 = getindex.(xp, 2)

    heatmap(axes..., data')
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
    global axes
    global points
    global default_res
    if res !== nothing && res !== default_res # recompute
        default_res = res
        axes, points = generateAxes(region; res=default_res)
    elseif axes === nothing # recompute
        axes, points = generateAxes(region; res=default_res)
    end
    return axes, points
end

function generateAxes(region; res=default_res)
    global axes = (:).(region.lb, res, region.ub)
    global points = collect.(Iterators.product(axes...))
    return axes, points
end

end
