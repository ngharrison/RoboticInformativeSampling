# pieces for the environment

module Environment

using Distributions

export Region, Map, GroundTruth, GaussGroundTruth, Peak, pointToIndex

struct Region
    lb # lower bounds
    ub # upper bounds
    obsMap # obstacle map
    gtMap # ground truth map
end

struct Map{T} <: AbstractMatrix{T}
    data::Matrix{T}
    res
end

# make a map function like a matrix
Base.size(m::Map) = size(m.data)
Base.IndexStyle(::Type{<:Map}) = IndexLinear()
Base.getindex(m::Map, i::Int) = m.data[i]
Base.setindex!(m::Map, v, i::Int) = m.data[i] = v

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

# helper method used with maps
pointToIndex(x, map) = CartesianIndex(Tuple(round.(Int, x ./ map.res) .+ 1))
indexToPoint(i, map) = (collect(Tuple(i)) .- 1) .* map.res

abstract type GroundTruth end

# ground truth struct for gaussian peaks
struct GaussGroundTruth <: GroundTruth
    peaks
end

function (groundTruth::GaussGroundTruth)(X)
    # produces ground-truth value(s) for a point or list of points
    # accepts a single vector, a vector of vectors, or a matrix of column vectors
    h_max = maximum(p.h for p in groundTruth.peaks)
    # create a probability distribution and divide by its own peak and the highest of all the peaks
    # this will cause the entire ground truth map to have a max value of 1
    return sum(p.h*pdf(p.distr, X)/pdf(p.distr, p.distr.μ)/h_max for p in groundTruth.peaks)
end

struct Peak
    distr
    h
end

Peak(μ, Σ, h) = Peak(MvNormal(μ, Σ), h)

end
