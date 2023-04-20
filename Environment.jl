module Environment

using Distributions

export Region, Map, imgToMap, pointToIndex, indexToPoint, res, GroundTruth,
GaussGroundTruth, Peak

"""
A general type for holding 2D data along with associated cell widths and
heights. It's main purpose is to handle the conversion between world coordinates
and grid indices internally. Accepts a 2-element vector as a coordinate pair.
Converting between the two representations treats rows as the first variable
(x-axis) and columns as the second (y-axis).

Also made to function like a built-in matrix directly by sub-typing and
implementing the base methods.

Fields:

    - data: matrix of data
    - lb: vector of lower bounds
    - ub: vector of upper bounds

Usage:

```julia
m = Map(matrix, lb, ub)
m(x) # returns the value at a single 2D point
m[i,j] # can also use as if it's just the underlying matrix
```
"""
struct Map{T1, T2} <: AbstractMatrix{T1}
    data::Matrix{T1}
    lb::Vector{T2}
    ub::Vector{T2}
end

"""
Takes a matrix in the format created from an image, re-formats it, and returns a
Map. Images view a matrix with its indexing top-down and left-right. Maps view a
matrix with its indexing left-right and bottom-up.
"""
imgToMap(img, lb, ub) = Map(permutedims(reverse(img, dims=1), (2,1)), lb, ub)

# make a map function like a matrix
Base.size(m::Map) = size(m.data)
Base.IndexStyle(::Type{<:Map}) = IndexLinear()
Base.getindex(m::Map, i::Int) = m.data[i]
Base.setindex!(m::Map, v, i::Int) = (m.data[i] = v)

function (map::Map)(x)
    # accepts a single vector
    index = pointToIndex(x, map)
    return map[index]
end

"""
A general container to hold data and metadata of the search region.

Fields:

    - occMap: occupancy map
    - gtMap: ground truth map
"""
struct Region
    occMap::Map{Bool}
    gtMap::Map{Float64}
end

# helper methods used with maps
res(map) = (map.ub .- map.lb) ./ (size(map) .- 1)
pointToIndex(x, map) = CartesianIndex(Tuple(round.(Int, (x .- map.lb) ./ res(map)) .+ 1))
indexToPoint(ci, map) = (collect(Tuple(ci)) .- 1) .* res(map) .+ map.lb

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
