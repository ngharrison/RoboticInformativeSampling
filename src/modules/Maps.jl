module Maps

using Distributions: MvNormal, pdf
using DocStringExtensions: SIGNATURES, TYPEDFIELDS

export Map, GaussGroundTruth, MultiMap, Peak,
       imgToMap, res, pointToCell, cellToPoint,
       Location, SampleInput, SampleOutput

const Location = Vector{Float64}
const SampleInput = Tuple{Location, Int}
const SampleOutput = Float64

"""
A general type for holding 2D data along with associated map bounds. It's main
purpose is to handle the conversion between world coordinates and grid indices
internally. Accepts a 2-element vector as a coordinate pair.  Converting between
the two representations treats rows as the first variable (x-axis) and columns
as the second (y-axis).

Also made to function like a built-in matrix directly by sub-typing and
implementing the base methods.

Fields:
$(TYPEDFIELDS)

Usage:

```julia
m = Map(matrix, lb, ub)
m(x) # returns the value at a single 2D point
m[i,j] # can also use as if it's just the underlying matrix
```
"""
struct Map{T1<:Real, M<:AbstractMatrix{T1}, T2<:Real} <: AbstractMatrix{T1}
    "matrix of data"
    data::M
    "vector of lower bounds, defaults to [0, 0]"
    lb::Vector{T2}
    "vector of upper bounds, defaults to [1, 1]"
    ub::Vector{T2}
end

function Base.show(io::IO, map::Map{T1}) where T1
    print(io, "Map{$T1} [$(map.lb), $(map.ub)]")
end
function Base.show(io::IO, ::MIME"text/plain", map::Map{T1}) where T1
    print(io, "Map{$T1} [$(map.lb), $(map.ub)]:\n")
    show(io, "text/plain", map.data)
end

Map(data::AbstractMatrix{<:Real}) = Map(data, [0.0, 0.0], [1.0, 1.0])

"""
Takes a matrix in the format created from an image, re-formats it, and returns a
Map. Images view a matrix with its indexing top-down and left-right. Maps view a
matrix with its indexing left-right and bottom-up.
"""
imgToMap(img, args...) = Map(permutedims(reverse(img, dims=1), (2,1)), args...)

# make a map behave like an array
Base.size(m::Map) = size(m.data)
Base.IndexStyle(::Type{<:Map}) = IndexLinear()
Base.getindex(m::Map, i::Integer) = m.data[i]
Base.setindex!(m::Map, v, i::Integer) = (m.data[i] = v)

"""
Function emits error if location is outside of map bounds.
"""
function checkBounds(x::Location, map::Map)
    all(map.lb .<= x .<= map.ub) || error("location $x is out of map bounds: ($(map.lb), $(map.ub))")
end

# accepts a single vector, returns a scalar
function (map::Map)(x::Location)
    checkBounds(x, map)
    map[pointToCell(x, map)]
end

"""
Generates a random point in the map. Returns the location and its value.
"""
function Base.rand(map::Map)
    x = map.lb .+ rand(2).*(map.ub .- map.lb)
    return x, map(x)
end

"""
Handles samples of the form (location, quantity) to give the value from the
right map. Internally a list of maps.

Constructor can take in a tuple or vector of Maps or each Map as a separate
argument.
"""
struct MultiMap{T1<:Real}
    maps::Tuple{Vararg{Map{T1}}}
end

function Base.show(io::IO, mmap::MultiMap{T1}) where T1
    print(io, "MultiMap{$T1}:")
    for map in mmap.maps
        print("\n\t", map)
    end
end

MultiMap(maps::Map...) = MultiMap(maps)
MultiMap(maps::AbstractVector{<:Map}) = MultiMap(Tuple(maps))

# make a multimap behave like an array
Base.keys(m::MultiMap) = keys(m.maps)
Base.length(m::MultiMap) = length(m.maps)
Base.iterate(m::MultiMap) = iterate(m.maps)
Base.iterate(m::MultiMap, i::Integer) = iterate(m.maps, i)
Base.Broadcast.broadcastable(m::MultiMap) = Ref(m) # don't broadcast
Base.IndexStyle(::Type{<:MultiMap}) = IndexLinear()
Base.getindex(m::MultiMap, i::Integer) = m.maps[i]

(mmap::MultiMap)(x::Location) = [mmap.maps[i](x) for i in eachindex(mmap)]
(mmap::MultiMap)(x::SampleInput) = mmap.maps[x[2]](x[1])

# helper methods used with maps
"""
Returns the resolution for each dimension of the given Map as a vector.
"""
res(map) = (map.ub .- map.lb) ./ (size(map) .- 1)

"""
Takes in a point in world-coordinates and a Map and returns a CartesianIndex for
the underlying matrix.
"""
pointToCell(x, map) = CartesianIndex(Tuple(round.(Int, (x .- map.lb) ./ res(map)) .+ 1))

"""
Takes in a CartesianIndex and a Map and returns a point in world-coordinates.
"""
cellToPoint(ci, map) = (collect(Tuple(ci)) .- 1) .* res(map) .+ map.lb


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
function (sampler::GaussGroundTruth)(X)
    h_max = maximum(p.h for p in sampler.peaks)
    return sum(p.h*pdf(p.distr, X)/pdf(p.distr, p.distr.μ)/h_max for p in sampler.peaks)
end

"""
Used within a GaussGroundTruth. Holds a 2D normal distribution and the
desired height of the peak.
"""
struct Peak
    distr
    h
end

"""
$(SIGNATURES)

Inputs:
- `μ`: the peak location (distribution mean)
- `Σ`: the peak width (distribution covariance)
- `h`: the peak height
"""
Peak(μ, Σ, h) = Peak(MvNormal(μ, Σ), h)

end
