module Kernels

using Statistics: I, mean
using AbstractGPs: with_lengthscale, SqExponentialKernel, IntrinsicCoregionMOKernel, CustomMean
using DocStringExtensions: TYPEDSIGNATURES

export singleKernel, multiKernel, slfmKernel, mtoKernel,
       fullyConnectedCovNum, fullyConnectedCovMat, manyToOneCovNum,
       manyToOneCovMat, initHyperparams, customKernel, multiMean

include("SLFMKernel.jl")
include("CustomKernel.jl")

## Mean and kernel stuff

multiMean(θ) = CustomMean(x->θ.μ[x[2]])

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

mtoKernel(θ) = IntrinsicCoregionMOKernel(kernel=with_lengthscale(SqExponentialKernel(), θ.ℓ^2),
                                           B=manyToOneCovMat(θ.σ))

slfmKernel(θ) = SLFMMOKernel(with_lengthscale.(SqExponentialKernel(), θ.ℓ.^2), θ.σ)

customKernel(θ) = CustomMOKernel(with_lengthscale.(SqExponentialKernel(), fullyConnectedCovMat(θ.ℓ)),
                                 fullyConnectedCovMat(θ.σ))

"""
$(TYPEDSIGNATURES)

Creates an output covariance matrix from an array of parameters by filling a lower
triangular matrix.

Inputs:
- `a`: parameter vector, must hold (N+1)*N/2 parameters, where N = number of
  outputs
"""
function fullyConnectedCovMat(a)

    N = floor(Int, sqrt(length(a)*2)) # (N+1)*N/2 in matrix

    # cholesky factorization technique to create a free-form covariance matrix
    # that is positive semidefinite
    L = [(v<=u ? a[u*(u-1)÷2 + v] : 0.0) for u in 1:N, v in 1:N]
    A = L*L' # lower triangular times upper

    return A + √eps()*I
end

fullyConnectedCovNum(num_outputs) = (num_outputs+1)*num_outputs÷2

"""
$(TYPEDSIGNATURES)

Creates an output covariance matrix from an array of parameters by filling the
first column and diagonal of a lower triangular matrix.

Inputs:
- `a`: parameter vector, must hold 2N-1 parameters, where N = number of
  outputs
"""
function manyToOneCovMat(a)

    # cholesky factorization technique to create a free-form covariance matrix
    # that is positive semidefinite
    N = (length(a)+1)÷2 # N on column + N-1 more on diagonal = 2N-1
    L = zeros(N,N) # will be lower triangular
    L[1,1] = a[1]
    for t=2:N
        L[t,1] = a[1 + t-1]
        L[t,t] = a[1 + 2*(t-1)]
    end
    A = L'*L # upper triangular times lower

    return A + √eps()*I
end

manyToOneCovNum(num_outputs) = 2*num_outputs - 1

"""
$(TYPEDSIGNATURES)

Creates the structure of hyperparameters for a MTGP and gives them initial values.
"""
function initHyperparams(X, Y_vals, bounds, N, ::typeof(multiKernel); kwargs...)
    n = fullyConnectedCovNum(N)
    σ = 0.5/sqrt(2) * ones(n)
    ℓ = mean(bounds.upper .- bounds.lower)
    return (; σ, ℓ, kwargs...)
end

"""
$(TYPEDSIGNATURES)

Creates the structure of hyperparameters for a MTGP and gives them initial values.
This is for a specialized quantity covariance matrix with separation.
"""
function initHyperparams(X, Y_vals, bounds, N, ::typeof(mtoKernel); kwargs...)
    n = manyToOneCovNum(N)
    σ = 0.5/sqrt(2) * ones(n)
    ℓ = mean(bounds.upper .- bounds.lower)
    return (; σ, ℓ, kwargs...)
end

"""
$(TYPEDSIGNATURES)

Creates the structure of hyperparameters for a SLFM and gives them initial values.
"""
function initHyperparams(X, Y_vals, bounds, N, ::typeof(slfmKernel); kwargs...)
    σ = 0.5/sqrt(2) * ones(N,N)
    ℓ = mean(bounds.upper .- bounds.lower) * ones(N)
    return (; σ, ℓ, kwargs...)
end

function initHyperparams(X, Y_vals, bounds, N, ::typeof(customKernel); kwargs...)
    n = fullyConnectedCovNum(N)
    σ = 0.5/sqrt(2) * ones(n)
    ℓ = mean(bounds.upper .- bounds.lower) * ones(n)
    return (; σ, ℓ, kwargs...)
end

end
