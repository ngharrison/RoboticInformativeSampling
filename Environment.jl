module Environment

using Distributions

export Region, Map, GroundTruth, GaussGroundTruth, Peak, pointToIndex, indexToPoint

"""
A general container to hold data and metadata of the search region.

Fields:

    - lb: lower bounds
    - ub: upper bounds
    - obsMap: obstacle map
    - gtMap: ground truth map
    - priorData: an array of already collected data
"""
struct Region
    lb
    ub
    obsMap
    gtMap
    prior_data
end

"""
A general type for holding 2D data along with associated cell widths and
heights. It's main purpose is to handle the conversion between world coordinates
and grid indices internally. Accepts a 2-element vector as a coordinate pair.
Converting between the two representations treats rows as the first variable
(x-axis) and columns as the second (y-axis).

Also made to function like a built-in matrix directly by sub-typing and
implementing the base methods.

Usage:

```julia
m = Map(matrix, grid_resolutions)
m(x) # returns the value at a single 2D point
m[i,j] # can also use as if it's just the underlying matrix
```
"""
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
    # accepts a single vector
    index = pointToIndex(x, map)
    return map[index]
end

# helper method used with maps
pointToIndex(x, map) = CartesianIndex(Tuple(round.(Int, x ./ map.res) .+ 1))
indexToPoint(i, map) = (collect(Tuple(i)) .- 1) .* map.res

abstract type GroundTruth end

"""
Struct/function for generating ground truth values from a linear combination of
gaussian peaks.

Usage:

```julia
GaussGroundTruth(peaks) # pass in a list of Peaks
```
"""
struct GaussGroundTruth <: GroundTruth
    peaks
end

"""
Produces ground-truth value(s) for a point or list of points. Accepts a single
vector, a vector of vectors, or a matrix of column vectors.

Each probability distribution component is divided by its own peak height and
the highest of all the peaks before being added into the total. This causes the
entire ground truth map to have a max value of (about) 1.
"""
function (groundTruth::GaussGroundTruth)(X)
    h_max = maximum(p.h for p in groundTruth.peaks)
    return sum(p.h*pdf(p.distr, X)/pdf(p.distr, p.distr.μ)/h_max for p in groundTruth.peaks)
end

"""
Used within a GaussGroundTruth. Holds a 2D normal distribution and the
desired height of the peak.

Usage: `Peak(μ, Σ, h)`

Inputs:

    - μ: the peak location (distribution mean)
    - Σ: the peak width (distribution covariance)
    - h: the peak height
"""
struct Peak
    distr
    h
end

Peak(μ, Σ, h) = Peak(MvNormal(μ, Σ), h)

end
