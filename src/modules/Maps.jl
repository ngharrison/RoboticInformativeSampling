module Maps

using Distributions: MvNormal, pdf
using DocStringExtensions: TYPEDSIGNATURES, TYPEDFIELDS

export Map, GaussGroundTruth, Peak, imgToMap, randomPoint,
       res, pointToCell, cellToPoint, generateAxes,
       ConstantRegion

"""
A general type for holding multidimensional data (usually a matrix) along with
associated dimension bounds. It's main purpose is to handle the conversion
between world coordinates and grid indices internally. Converting between the
two representations treats rows as the first variable (x-axis), columns as
the second (y-axis), and so on.

It's typical use is to act as a 2D map of some value that can be sampled.

Also made to function directly like a built-in N-dimensional array by sub-typing
and implementing the base methods.

Fields:
$(TYPEDFIELDS)
"""
struct Map{T1<:Any, N<:Any, A<:AbstractArray{T1, N}, T2<:Real} <: AbstractArray{T1, N}
    "N-dimensional array of data"
    data::A
    "vector of lower bounds, defaults to zeros"
    lb::Vector{T2}
    "vector of upper bounds, defaults to ones"
    ub::Vector{T2}

    function Map(data, lb, ub)
        length(lb) == length(ub) == ndims(data) ||
            throw(DimensionMismatch("lengths of bounds don't match data dimensions"))
        new{eltype(data), ndims(data), typeof(data), eltype(lb)}(data, lb, ub)
    end
end

Map(data::AbstractArray{<:Any}) = Map(data, zeros(ndims(data)), ones(ndims(data)))

"""
Method accepts a single vector (the location), returns a scalar (the value at
that point).

# Examples
```julia
data = reshape(1:25, 5, 5)
lb = [0.0, 0.0]
ub = [1.0, 1.0]
m = Map(data, lb, ub)
m2 = Map(data) # bounds will be zero to one

x = [.2, .75]
val = m(x) # returns the value at a single 2D point
val2 = m[1,4] # can also use as if it's just the underlying matrix
```
"""
function (map::Map)(x)
    checkBounds(x, map)
    map[pointToCell(x, map)]
end

# make a map behave like an array
Base.size(m::Map) = size(m.data)
Base.IndexStyle(::Type{<:Map}) = IndexLinear()
Base.getindex(m::Map, i::Integer) = m.data[i]
Base.setindex!(m::Map, v, i::Integer) = (m.data[i] = v)

# change display
function Base.show(io::IO, map::Map{T1}) where T1
    print(io, "Map{$T1} [$(map.lb), $(map.ub)]")
end
function Base.show(io::IO, ::MIME"text/plain", map::Map{T1}) where T1
    print(io, "Map{$T1} [$(map.lb), $(map.ub)]:\n")
    show(io, "text/plain", map.data)
end

"""
$(TYPEDSIGNATURES)

Generates a random point in the map. Returns the location and its value.

# Examples
```julia
data = reshape(1:25, 5, 5)
map = Map(data)
rand(map)
```
"""
function Base.rand(map::Map)
    x = randomPoint(map)
    return x, map(x)
end

function randomPoint(map::Map)
    return map.lb .+ rand(ndims(map)).*(map.ub .- map.lb)
end

# helper methods used with maps
"""
Takes a matrix in the format created from an image, re-formats it, and returns a
Map. Images view a matrix with its indexing top-down and left-right. Maps view a
matrix with its indexing left-right and bottom-up.

# Examples
```julia
using DelimitedFiles: readdlm

image = readdlm(file_name, ',')
lb = [0.0, 0.0]; ub = [1.0, 1.0]
map = imgToMap(image, lb, ub)
map = imgToMap(image) # or auto bounds
```
"""
imgToMap(img, args...) = Map(permutedims(reverse(img, dims=1), (2,1)), args...)

"""
$(TYPEDSIGNATURES)

Function emits error if location is outside of map bounds.

# Examples
```julia
x = [.2, .75]
data = reshape(1:25, 5, 5)
map = Map(data)
checkBounds(x, map) # no error thrown
```
"""
function checkBounds(x, map::Map)
    length(x) == ndims(map) ||
        throw(DomainError(x, "length doesn't match map dimensions: $(ndims(map))"))
    all(map.lb .<= x .<= map.ub) ||
        throw(DomainError(x, "out of map bounds: ($(map.lb), $(map.ub))"))
end

"""
$(TYPEDSIGNATURES)

Returns the resolution for each dimension of the given Map as a vector.
"""
res(map) = (map.ub .- map.lb) ./ (size(map) .- 1)

"""
$(TYPEDSIGNATURES)

Takes in a point in world-coordinates and a Map and returns a CartesianIndex for
the underlying array.
"""
pointToCell(x, map) = CartesianIndex(Tuple(round.(Int, (x .- map.lb) ./ res(map)) .+ 1))

"""
$(TYPEDSIGNATURES)

Takes in a CartesianIndex and a Map and returns a point in world-coordinates.
"""
cellToPoint(ci, map) = (collect(Tuple(ci)) .- 1) .* res(map) .+ map.lb

"""
$(TYPEDSIGNATURES)

Method to generate the x, y, etc. axes and points of a Map. Useful for plotting.
"""
function generateAxes(map)
    axes = range.(map.lb, map.ub, size(map))
    points = collect.(Iterators.product(axes...))
    return axes, points
end


abstract type GroundTruth end

"""
Struct/function for generating ground truth values from a linear combination of
gaussian peaks.

# Examples
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
function (ggt::GaussGroundTruth)(X)
    h_max = maximum(p.h for p in ggt.peaks)
    return sum(p.h*pdf(p.distr, X)/pdf(p.distr, p.distr.μ)/h_max for p in ggt.peaks)
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
$(TYPEDSIGNATURES)

Inputs:
- `μ`: the peak location (distribution mean)
- `Σ`: the peak width (distribution covariance)
- `h`: the peak height
"""
Peak(μ, Σ, h) = Peak(MvNormal(μ, Σ), h)

"""
$(TYPEDSIGNATURES)

Type that can be called to return one value when within its bounds and another
when without. Pass it a multi-dimensional point, like a location in space.
"""
struct ConstantRegion{T1, T2}
    "single value that this type returns when within bounds"
    in_bounds_value::T1
    "single value that this type returns when out of bounds"
    out_of_bounds_value::T1
    "vector of lower bounds, defaults to zeros"
    lb::Vector{T2}
    "vector of upper bounds, defaults to ones"
    ub::Vector{T2}
end

function (cr::ConstantRegion)(x)
    all(cr.lb .<= x .<= cr.ub) ?
        cr.in_bounds_value :
        cr.out_of_bounds_value
end

end
