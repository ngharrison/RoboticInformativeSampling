# pieces for the environment

module Environment

using LinearAlgebra
using Distributions

export Region, Map, GT, GaussGT, Peak

struct Region
    lb # lower bounds
    ub # upper bounds
    obsMap # obstacle map
    gtMap # ground truth map
end

struct Map
    data
    res
end

# helper method used with maps
pointToIndex(x, res) = round.(Int, x ./ res) .+ 1

function (map::Map)(x)
    # produces a ground-truth value for a point
    # accepts a single vector
    index = pointToIndex(x, map.res)
    return map.data[index...]
end

function (map::Map)()
    # if called with no points, return the entire map
    return map.data
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
