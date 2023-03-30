# pieces for the environment

module Environment

using LinearAlgebra
using Distributions

export GT, GaussGT, Peak, Region

struct Region
    lb # lower bounds
    ub # upper bounds
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

# for future
# struct GeneralGT <: GT
#     map
#     res
# end

# function (gt::GeneralGT)(X)
#     # produces ground-truth value(s) for a point or list of points
#     # accepts a single vector, a vector of vectors, or a matrix of column vectors
#     indices = [round.(Int, x ./ gt.res).+1 for x in X]
#     return [gt.map[ind...] for ind in indices]
# end

end
