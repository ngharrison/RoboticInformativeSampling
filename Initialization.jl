module Initialization

using LinearAlgebra
using Distributions

export GT, initializeRegion, createGT

struct Region
    lb # lower bounds
    ub # upper bounds
end

# ground truth struct
struct GT
    μ # expected value
    Σ # covariance matrix
    distr # distribution that generates the values
end

(gt::GT)(X) = pdf(gt.distr, X)./pdf(gt.distr, gt.μ)

function createGT(μ=[0.3, 0.3], Σ=0.03*I)
    # returns a function that produces ground-truth value(s) for a point or list of points
    # accepts a single vector, a vector of vectors, or a matrix of column vectors
    # μ = [[0.3, 0.3], [0.6, 0.7]]
    # Σ = [0.03*I, 0.01*I]
    distr = MvNormal(μ, Σ)
    return GT(μ, Σ, distr)
end

function initializeRegion()
    lb = [0, 0]
    ub = [1, 1]
    # x3 = zeros(n)
    return Region(lb, ub)
end

end
