module BeliefModels

using LinearAlgebra
using AbstractGPs
using StatsFuns
using Optim
using ParameterHandling
using DocStringExtensions

using Environment

export BeliefModel, generateBeliefModel, fullyConnectedCovMat

"""
Belief model struct and function.

Designed on top of a Gaussian Process for 2D inputs.
"""
struct BeliefModel
    gp
    θ
end

"""
Inputs:

    - X: a single point or an array of multiple points

Outputs:

    - μ, σ: a pair of expected value(s) and uncertainty(s) for the given point(s)
"""
function (beliefModel::BeliefModel)(x::Vector{<:Real})
    pred = only(marginals(beliefModel.gp([(x, 1)])))
    return pred.μ, pred.σ
end

function (beliefModel::BeliefModel)(X::Vector{<:Vector{<:Real}})
    pred = marginals(beliefModel.gp(tuple.(X, 1)))
    μ = getfield.(pred, :μ)
    σ = getfield.(pred, :σ)
    return μ, σ
end

"""
$SIGNATURES

Creates and returns a new belief model containing a GP. The GP is trained and
conditioned on the given samples.
"""
function generateBeliefModel(samples, occMap)
    # set up training data
    X = getfield.(samples, :x)
    Y = getfield.(samples, :y)

    # number of outputs
    T = maximum(last, X)
    n = (T+1)*T÷2 # fullyConnectedCovMat
    # n = 2*T - 1 # manyToOneCovMat

    # set up hyperparameters
    σ = (length(Y)>1 ? std(Y) : 0.5)/sqrt(2) * ones(n)
    a = mean(mean([occMap.lb, occMap.ub]))
    ℓ = length(X)==1 ? a : a/length(X) + mean(std(first.(X)))*(1-1/length(X))
    σn = 0.001
    θ0 = (; σ, ℓ, σn)

    θ0_flat, unflatten = value_flatten(θ0)
    k = multiKernel
    loss_flat = lossFunction(X, Y, k) ∘ unflatten

    # optimize hyperparameters (train)
    opt = optimize(loss_flat, θ0_flat, LBFGS())
    θ = unflatten(opt.minimizer)

    # produce optimized gp belief model
    f = GP(k(θ)) # prior gp
    fx = f(X, θ.σn^2+√eps())
    f_post = posterior(fx, Y) # gp conditioned on training samples
    return BeliefModel(f_post, θ)
end

"""
$SIGNATURES

This function creates the loss function for training the GP. The negative log
marginal likelihood is used.
"""
function lossFunction(X, Y, k)
    # returns a function for the negative log marginal likelihood
    θ -> begin
        try
            f = GP(k(θ))
            fx = f(X, θ.σn^2+√eps()) # eps to prevent numerical issues
            return -logpdf(fx, Y)
        catch e
            # for PosDefException
            # this seems to happen when θ.σ is extremely large and θ.ℓ is
            # much bigger than the search region dimensions
            println(); println("Error: $e"); @show(θ, X, Y); println()

            f = GP(k(θ))
            fx = f(X, θ.σn^2+√eps()+1e-1*θ.σ) # fix by making diagonal a little bigger
            return -logpdf(fx, Y)
        end
    end
end


## Kernel stuff

"""
$SIGNATURES

A simple squared exponential kernel for the GP with parameters θ.
"""
singleKernel(θ) = θ.σ^2 * with_lengthscale(SqExponentialKernel(), θ.ℓ^2)

"""
$SIGNATURES

A multi-task GP kernel, a variety of multi-output GP kernel based on the
Intrinsic Coregionalization Model with a Squared Exponential base kernel and an
output matrix formed from a lower triangular matrix.
"""
multiKernel(θ) = IntrinsicCoregionMOKernel(kernel=with_lengthscale(SqExponentialKernel(), θ.ℓ^2),
                                           B=fullyConnectedCovMat(θ.σ))

function fullyConnectedCovMat(a)
# Creates an output covariance matrix from an array of parameters
# Fills a lower triangular matrix
# a must hold (T+1)*T/2 parameters, where T = number of outputs

    T = floor(Int, sqrt(length(a)*2)) # (T+1)*T/2 in matrix

    # cholesky factorization technique to create a free-form covariance matrix
    # that is positive semidefinite
    L = zeros(T,T) # will be lower triangular
    for u in 1:T
        for v in 1:u
            L[u,v] = a[u*(u-1)÷2 + v]
        end
    end
    # A = L'*L # upper triangular times lower
    A = L*L' # lower triangular times upper

    return A + √eps()*I
end


end
