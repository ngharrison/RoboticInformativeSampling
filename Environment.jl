# pieces for the environment

module Environment

using LinearAlgebra
using Distributions
using Statistics
using Graphs

export Region, Map, GT, GaussGT, Peak, pathCost

struct Region
    lb # lower bounds
    ub # upper bounds
    obsMap # obstacle map
    gtMap # ground truth map
    graph # graph for path finding
    graph_weights
    vmap # translates vertices in graph
end

function Region(lb, ub, obsMap, gtMap)
    # create a graph for path finding
    graph = grid(size(obsMap.data))
    li = LinearIndices(obsMap.data)
    m, n = size(obsMap.data)
    for i in 1:m, j in 1:n
        i > 1 && j > 1 && add_edge!(graph, li[i,j], li[i-1,j-1])
        i > 1 && j < n && add_edge!(graph, li[i,j], li[i-1,j+1])
        i < m && j > 1 && add_edge!(graph, li[i,j], li[i+1,j-1])
        i < m && j < n && add_edge!(graph, li[i,j], li[i+1,j+1])
    end

    # give diagonal paths correct distance in weight matrix
    graph_weights = ones(size(graph))
    ci = CartesianIndices(obsMap.data)
    for i in 1:size(graph_weights, 1), j in 1:size(graph_weights, 2)
        diff = ci[i] - ci[j]
        if abs(diff[1]) == 1 && abs(diff[2]) == 1
            graph_weights[i,j] = √2
        end
    end

    # remove all vertices in collision
    indices = findall(vec(obsMap.data))
    vmap = rem_vertices!(graph, indices, keep_order=true)

    Region(lb, ub, obsMap, gtMap, graph, graph_weights, vmap)
end

struct Map
    data
    res
end

# helper method used with maps
pointToIndex(x, map) = CartesianIndex(Tuple(round.(Int, x ./ map.res) .+ 1))

function (map::Map)(x)
    # produces a ground-truth value for a point
    # accepts a single vector
    index = pointToIndex(x, map)
    return map.data[index]
end

function (map::Map)()
    # if called with no points, return the entire map
    return map.data
end

function getGraphIndex(x, region)
    i = LinearIndices(region.obsMap.data)[pointToIndex(x, region.obsMap)]
    return findfirst(==(i), region.vmap)
end

function pathCost(x1, x2, region)
    # if either point is within an obstacle, just return infinity
    any(region.obsMap.([x1, x2])) && return Inf

    # calculate cost
    s, t = getGraphIndex.((x1, x2), Ref(region))
    path = a_star(region.graph, s, t, region.graph_weights, v->norm(t.-v))
    return length(path)*mean(region.obsMap.res)
end


abstract type GT end

# ground truth struct for gaussian peaks
struct GaussGT <: GT
    peaks
end

function (gt::GaussGT)(X)
    # produces ground-truth value(s) for a point or list of points
    # accepts a single vector, a vector of vectors, or a matrix of column vectors
    h_max = maximum(p.h for p in gt.peaks)
    # create a probability distribution and divide by its own peak and the highest of all the peaks
    # this will cause the entire GT map to have a max value of 1
    return sum(p.h*pdf(p.distr, X)/pdf(p.distr, p.distr.μ)/h_max for p in gt.peaks)
end

struct Peak
    distr
    h
end

Peak(μ, Σ, h) = Peak(MvNormal(μ, Σ), h)

end
