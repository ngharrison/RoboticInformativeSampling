"""
Handles samples of the form (location, quantity) to give the value from the
right map. Internally a tuple of GridMaps.

Constructor can take in a tuple or vector of GridMaps or each GridMap as a separate
argument.

# Examples
```julia
ss = GridMapsSampler(GridMap(zeros(5, 5)), GridMap(ones(5, 5)))

loc = [.2, .75]
ss(loc) # result: [0, 1]
ss((loc, 2)) # result: 1
```
"""
struct GridMapsSampler{T1<:Real}
    maps::Tuple{Vararg{GridMap{T1}}}
end

GridMapsSampler(maps::GridMap...) = GridMapsSampler(maps)
GridMapsSampler(maps::AbstractVector{<:GridMap}) = GridMapsSampler(Tuple(maps))

(ss::GridMapsSampler)(loc::Location) = [map(loc) for map in ss]
(ss::GridMapsSampler)((loc, q)::SampleInput) = ss[q](loc)

# make it behave like a tuple
Base.keys(m::GridMapsSampler) = keys(m.maps)
Base.length(m::GridMapsSampler) = length(m.maps)
Base.iterate(m::GridMapsSampler) = iterate(m.maps)
Base.iterate(m::GridMapsSampler, i::Integer) = iterate(m.maps, i)
Base.Broadcast.broadcastable(m::GridMapsSampler) = Ref(m) # don't broadcast
Base.IndexStyle(::Type{<:GridMapsSampler}) = IndexLinear()
Base.getindex(m::GridMapsSampler, i::Integer) = m.maps[i]

# change display
function Base.show(io::IO, ss::GridMapsSampler{T1}) where T1
    print(io, "GridMapsSampler{$T1}:")
    for map in ss
        print("\n\t", map)
    end
end
