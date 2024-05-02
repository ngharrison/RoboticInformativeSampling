module Kernels

using Statistics: I, mean
using AbstractGPs: with_lengthscale, SqExponentialKernel, IntrinsicCoregionMOKernel, CustomMean
using DocStringExtensions: TYPEDSIGNATURES

export multiMeanAve, singleKernel, multiKernel, fullyConnectedCovNum,
       slfmKernel, fullyConnectedCovMat, manyToOneCovNum, manyToOneCovMat,
       initHyperparams

include("SLFMKernel.jl")

## Mean and kernel stuff

"""
$(TYPEDSIGNATURES)

Creates the structure of hyperparameters for a MTGP and gives them initial values.
"""
function initHyperparams(X, Y_vals, lb, ub, ::typeof(multiKernel); kwargs...)
    T = maximum(last, X) # number of outputs
    n = fullyConnectedCovNum(T)
    # NOTE may change to all just 0.5
    # σ = (length(Y_vals)>1 ? std(Y_vals) : 0.5)/sqrt(2) * ones(n)
    # a = mean(ub .- lb)
    # ℓ = length(X)==1 ? a : a/length(X) + mean(std(first.(X)))*(1-1/length(X))
    σ = 0.5/sqrt(2) * ones(n)
    ℓ = mean(ub .- lb)
    return (; σ, ℓ, kwargs...)
end

"""
$(TYPEDSIGNATURES)

Creates the structure of hyperparameters for a SLFM and gives them initial values.
"""
function initHyperparams(X, Y_vals, lb, ub, ::typeof(slfmKernel); kwargs...)
    T = maximum(last, X) # number of outputs
    # NOTE may change to all just 0.5
    # σ = (length(first.(Y_vals))>1 ? std(first.(Y_vals)) : 0.5)/sqrt(2) * ones(n)
    # a = mean(ub .- lb)
    # ℓ = (length(X)==1 ? a : a/length(X) + mean(std(first.(X)))*(1-1/length(X))) * ones(T)
    σ = 0.5/sqrt(2) * ones(T,T)
    ℓ = mean(ub .- lb) * ones(T)
    return (; σ, ℓ, kwargs...)
end

function multiMeanAve(X, Y)
    # calculate means
    T = maximum(last, X)
    mean_vals = zeros(T)
    nums = zeros(Int, T)
    for ((_, q), y) in zip(X, Y)
        mean_vals[q] += y
        nums[q] += 1
    end
    mean_vals ./= nums

    # return function inside CustomMean
    CustomMean(x->mean_vals[x[2]])
end

"""
$(TYPEDSIGNATURES)

A simple squared exponential kernel for the GP with parameters `θ`.

This function creates the kernel function used within the GP.
"""
singleKernel(θ) = θ.σ^2 * with_lengthscale(SqExponentialKernel(), θ.ℓ^2)

"""
$(TYPEDSIGNATURES)

A multi-task GP kernel, a variety of multi-output GP kernel based on the
Intrinsic Coregionalization Model with a Squared Exponential base kernel and an
output matrix formed from a lower triangular matrix.

This function creates the kernel function used within the GP.
"""
multiKernel(θ) = IntrinsicCoregionMOKernel(kernel=with_lengthscale(SqExponentialKernel(), θ.ℓ^2),
                                           B=fullyConnectedCovMat(θ.σ))

fullyConnectedCovNum(num_outputs) = (num_outputs+1)*num_outputs÷2

slfmKernel(θ) = SLFMMOKernel(with_lengthscale.(SqExponentialKernel(), θ.ℓ.^2), θ.σ)

"""
$(TYPEDSIGNATURES)

Creates an output covariance matrix from an array of parameters by filling a lower
triangular matrix.

Inputs:
- `a`: parameter vector, must hold (T+1)*T/2 parameters, where T = number of
  outputs
"""
function fullyConnectedCovMat(a)

    T = floor(Int, sqrt(length(a)*2)) # (T+1)*T/2 in matrix

    # cholesky factorization technique to create a free-form covariance matrix
    # that is positive semidefinite
    L = [(v<=u ? a[u*(u-1)÷2 + v] : 0.0) for u in 1:T, v in 1:T]
    A = L*L' # lower triangular times upper

    return A + √eps()*I
end

manyToOneCovNum(num_outputs) = 2*num_outputs - 1

"""
$(TYPEDSIGNATURES)

Creates an output covariance matrix from an array of parameters by filling the
first column and diagonal of a lower triangular matrix.

Inputs:
- `a`: parameter vector, must hold 2T-1 parameters, where T = number of
  outputs
"""
function manyToOneCovMat(a)

    # cholesky factorization technique to create a free-form covariance matrix
    # that is positive semidefinite
    T = (length(a)+1)÷2 # T on column + T-1 more on diagonal = 2T-1
    L = zeros(T,T) # will be lower triangular
    L[1,1] = a[1]
    for t=2:T
        L[t,1] = a[1 + t-1]
        L[t,t] = a[1 + 2*(t-1)]
    end
    A = L'*L # upper triangular times lower

    return A + √eps()*I
end

end
