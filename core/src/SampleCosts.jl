module SampleCosts

using LinearAlgebra: norm
using Statistics: mean
using DocStringExtensions: TYPEDSIGNATURES

using ..Maps: pointToCell, cellToPoint, res, getBounds
using ..Paths: PathCost

export SampleCost, values, DistProx,
       LogNormed, MIPT, EIGF,
       DistScaledEIGF, DistLogEIGF

abstract type SampleCost end

"""
$(TYPEDSIGNATURES)

Cost to take a new sample at a location. This is a fallback method that
calculates a simple linear combination of all the values of a SampleCost.

Has the form:
``\\mathrm{cost} = - w_{1} μ - w_{2} σ + w_{3} τ + w_{4} D``
"""
function (sc::SampleCost)(loc)
    # sc.occupancy(loc) && return Inf
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

end
