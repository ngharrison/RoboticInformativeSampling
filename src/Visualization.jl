module Visualization

using Plots: plot, heatmap, scatter!, @layout, mm, grid
using DocStringExtensions: SIGNATURES

using Environment: GroundTruth, Map, res
using BeliefModels: BeliefModel
using SampleCosts: SampleCost

# placeholders to avoid recomputing
axes = nothing
points = nothing

# default properties for all plots
default_res = [0.01, 0.01]
default_size = (1200, 800)
default_markersize = 4
default_margin = 5mm
sample_color = :green
new_sample_color = :red

"""
$SIGNATURES

Main method to visualize the current state of a search. Generates all the other
visuals through their respective methods and lays them out in a grid. Currently
shows the belief model, the ground truth, and the obstacles.

Arguments pass through to the sub-methods that need them. res is the grid
resolution when plotting continuous-valued functions and defaults to $default_res.

If no ground truth is available, it is not plotted.
"""
function visualize(md, beliefModel::BeliefModel, sampleCost, samples; quantity)
    a = visualize(beliefModel, samples, md.occupancy, quantity)
    b = plot(legend=false, grid=false, foreground_color_subplot=:white) # blank plot
    # TODO this will need to be updated to test for actual types
    if hasmethod(getindex, Tuple{typeof(md.groundTruth), Integer})
        # we actually have ground truth
        b = visualize(md.groundTruth[quantity], "Ground Truth")
    end
    c = visualize(sampleCost, samples, md.occupancy)
    l = @layout [a ; b c]
    plot(a, b, c, layout=l, size=default_size, margin=default_margin)
end

"""
$SIGNATURES

Method to show a ground truth map and up to three other prior data maps.
Pass each map in as its own argument.
"""
function visualize(maps::Map...;
                   titles=["Ground Truth"; ["Prior $i" for i in 1:length(maps)-1]],
                   samples=[])
    plot(
        visualize.(maps, titles; samples, clim=extrema(maps[1]))...,
        layout=grid(2,2),
        size=default_size,
        margin=default_margin
    )
end

## functions to visualize individual pieces

"""
$SIGNATURES

Method to show any Map data.
"""
function visualize(map::Map, title="Map"; samples=[], clim=nothing)
    axes = range.(map.lb, map.ub, size(map))
    heatmap(axes..., map';
            xlabel="x1",
            ylabel="x2",
            title,
            clim
            )
    scatter!(first.(samples), last.(samples);
             label="Samples",
             color=sample_color,
             markersize=default_markersize)
end

"""
$SIGNATURES

Method to show ground truth data.
"""
function visualize(groundTruth::GroundTruth, map)
    axes, points = getAxes(map)
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
function visualize(beliefModel::BeliefModel, samples, occupancy, quantity)
    axes, points = getAxes(occupancy)
    dims = Tuple(length.(axes))
    μ, σ = beliefModel(tuple.(vec(points), quantity))
    pred_map = reshape(μ, dims)
    err_map = reshape(σ, dims)

    # blocked points
    pred_map[occupancy] .= NaN
    err_map[occupancy] .= NaN

    xp = first.(getfield.(samples, :x))
    x1 = getindex.(xp, 1)
    x2 = getindex.(xp, 2)

    p1 = heatmap(axes..., pred_map')
    scatter!(x1[begin:end-1], x2[begin:end-1];
             xlabel="x1",
             ylabel="x2",
             title="GP Mean",
             legend=nothing,
             color=sample_color,
             markersize=default_markersize)
    scatter!([x1[end]], [x2[end]];
             color=new_sample_color,
             markersize=default_markersize)

    p2 = heatmap(axes..., err_map')
    scatter!(x1[begin:end-1], x2[begin:end-1];
             xlabel="x1",
             ylabel="x2",
             title="GP Std",
             legend=nothing,
             color=sample_color,
             markersize=default_markersize)
    scatter!([x1[end]], [x2[end]];
             color=new_sample_color,
             markersize=default_markersize)

    plot(p1, p2)
end


"""
$SIGNATURES

Method to show sample cost values.
"""
function visualize(sampleCost, samples, occupancy)
    isnothing(sampleCost) && return plot()

    axes, points = getAxes(occupancy)
    data = -sampleCost.(points)

    # blocked points
    data[occupancy] .= NaN

    xp = first.(getfield.(samples, :x))
    x1 = getindex.(xp, 1)
    x2 = getindex.(xp, 2)

    heatmap(axes..., data')
    scatter!(x1[begin:end-1], x2[begin:end-1];
             xlabel="x1",
             ylabel="x2",
             title="Obj Function",
             legend=nothing,
             color=sample_color,
             markersize=default_markersize)
    scatter!([x1[end]], [x2[end]], color=new_sample_color, markersize=default_markersize)
end

"""
$SIGNATURES

Method to get the x and y plotting axes. This (re)generates them only if needed
and saves them into global module variables for future use.
"""
function getAxes(map)
    global axes
    global points
    global default_res
    if res(map) !== default_res # recompute
        default_res = res(map)
        axes, points = generateAxes(map)
    elseif axes === nothing # recompute
        axes, points = generateAxes(map)
    end
    return axes, points
end

"""
$SIGNATURES

Method to generate the x and y plotting axes.
"""
function generateAxes(map)
    global axes = (:).(map.lb, res(map), map.ub)
    global points = collect.(Iterators.product(axes...))
    return axes, points
end

end
