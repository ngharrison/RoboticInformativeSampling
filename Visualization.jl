module Visualization

using Plots
using DocStringExtensions

using Environment
using BeliefModels
using Samples

export visualize

# placeholders to avoid recomputing
axes = nothing
points = nothing

default_res = [0.01, 0.01]
sample_color = :green
new_sample_color = :red

"""
$SIGNATURES

Main method to visualize everything. Generates all the other visuals through
their respective methods and lays them out in a grid. Currently shows the belief
model, the ground truth, and the obstacles.

Arguments pass through to the sub-methods that need them. res is the grid
resolution plotting continuous-valued functions and defaults to $default_res.
"""
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

"""
$SIGNATURES

Method to show any Map data.
"""
function visualize(map::Map, region; title="Map")
    axes = (:).(region.lb, map.res, region.ub)
    heatmap(axes..., map';
            xlabel="x1",
            ylabel="x2",
            title
            )
end

"""
$SIGNATURES

Method to show ground truth data.
"""
function visualize(groundTruth::GroundTruth, region; res=default_res)
    axes, points = getAxes(region; res)
    data = groundTruth(points)
    heatmap(axes..., data';
            xlabel="x1",
            ylabel="x2",
            title="Ground Truth"
            )
end

"""
$SIGNATURES

Method to show belief model values of mean and standard deviation and the sample
locations that they were generated from. Shows two plots side-by-side.
"""
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


"""
$SIGNATURES

Method to show sample cost values.
"""
function visualize(sampleCost::SampleCost, samples, region; res=default_res)
    axes, points = getAxes(region; res)
    data = -sampleCost.(points)

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

"""
$SIGNATURES

Method to get the x and y plotting axes. This (re)generates them only if needed
and saves them into global module variables for future use.
"""
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

"""
$SIGNATURES

Method to generate the x and y plotting axes.
"""
function generateAxes(region; res=default_res)
    global axes = (:).(region.lb, res, region.ub)
    global points = collect.(Iterators.product(axes...))
    return axes, points
end

end
