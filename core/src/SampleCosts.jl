"""
This module holds a variety of SampleCost functions used by [Samples.jl](@ref) in
selecting a new sample location.

Main public types and functions:
$(EXPORTS)
"""
module SampleCosts

using LinearAlgebra: norm
using Statistics: mean
using DocStringExtensions: TYPEDEF, TYPEDSIGNATURES, EXPORTS

using ..Maps: pointToCell, cellToPoint, res, getBounds
using ..Paths: PathCost

export SampleCost, values, DistProx,
       LogNormed, MIPT, EIGF, OnlyVar,
       DistScaledEIGF, DistLogEIGF,
       DerivVar, DistScaledDerivVar,
       InfoGain, VarTrace, LogLikelihood,
       LogLikelihoodFull

abstract type SampleCost end

"""
$(TYPEDSIGNATURES)

Cost to take a new sample at a location. This is a fallback method that
calculates a simple linear combination of all the values of a SampleCost.

Has the form:
``\\mathrm{cost} = - w_{1} μ - w_{2} σ + w_{3} τ + w_{4} D``
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
