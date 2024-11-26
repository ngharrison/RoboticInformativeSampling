"""
This module holds a variety of SampleCost functions used by [Samples.jl](@ref)
in selecting a new sample location. The purpose is to pick the location that
will minimize the given function.

Each sample cost function in this module is a subtype of the abstract SampleCost
type. Their common interface consists of two functions:
- `values(sampleCost, location)`: returns the values of the terms (μ, σ, τ, P);\\
                                  this is typically what each subtype will override;\\
                                  in all of these, μ = belief model mean, σ = belief model std,
                                  τ = travel distance, P = proximity value;\\
                                  all of these are for the specific location
- `sampleCost(location)`: the actual sample cost at the location; it has a
                          default method explained in [SampleCost](@ref);\\
                          for the equations the location is denoted ``x``

Many of these were experiment cost functions and aren't recommended. The main
ones recommended for use, in order, are
- [DistScaledEIGF](@ref)
- [EIGF](@ref)
- [MIPT](@ref)
- [OnlyVar](@ref)

Others can be useful if one wants to do some more experimentation. Note that all
these functions are currently hardcoded to use the first quantity as the
objective quantity unless otherwise stated. Unless explicitly used, the distance
value is generally used to make sure unreachable locations are forbidden (their
value will be Inf).

Main public types and functions:
$(EXPORTS)
"""
module SampleCosts

using LinearAlgebra: norm
using Statistics: mean
using DocStringExtensions: TYPEDEF, TYPEDSIGNATURES, EXPORTS

using AStarGridSearch: PathCost
using GridMaps: pointToCell, cellToPoint, res, getBounds

export SampleCost, values, DistProx,
       LogNormed, MIPT, EIGF, OnlyVar,
       DistScaledEIGF, DistLogEIGF,
       DerivVar, DistScaledDerivVar,
       InfoGain, VarTrace, LogLikelihood,
       LogLikelihoodFull

"""
$(TYPEDSIGNATURES)

Typically construct a SampleCost through
`SampleCostType(occupancy, samples, beliefModel, quantities, weights)`

A pathCost is constructed automatically from the other arguments.

This object can then be called to get the cost of sampling at a location:
sampleCost(x)
"""
abstract type SampleCost end

"""
$(TYPEDSIGNATURES)

Returns the values to be used to calculate the sample cost (belief mean,
standard deviation, travel distance, sample proximity).

Each concrete subtype of SampleCost needs to implement.

This can be a useful function to inspect values during optimization.
"""
function values(sc::SampleCost, loc) end

"""
$(TYPEDSIGNATURES)

Cost to take a new sample at a location ``x``. This is a fallback method that
calculates a simple linear combination of all the values of a SampleCost.

Has the form:
```math
C(x) = - w_1 \\, μ(x) - w_2 \\, σ(x) + w_3 \\, τ(x) + w_4 \\, P(x)
```
"""
function (sc::SampleCost)(loc)
    sc.occupancy(loc) && return Inf
    vals = values(sc, loc)
    return sum(w*v for (w,v) in zip(sc.weights, vals))
end

include("sc_types/DistProx.jl")
include("sc_types/LogNormed.jl")
include("sc_types/MIPT.jl")
include("sc_types/EIGF.jl")
include("sc_types/DistScaledEIGF.jl")
include("sc_types/DistLogEIGF.jl")
include("sc_types/MEPE.jl")
include("sc_types/DerivVar.jl")
include("sc_types/OnlyVar.jl")
include("sc_types/DistScaledDerivVar.jl")
include("sc_types/InfoGain.jl")
include("sc_types/VarTrace.jl")
include("sc_types/LogLikelihood.jl")
include("sc_types/LogLikelihoodFull.jl")

end
