module BeliefModels

using AbstractGPs
using StatsFuns
using Optim
using ParameterHandling
using DocStringExtensions

export BeliefModel, generateBeliefModel

"""
Belief model struct and function.

Designed on top of a Gaussian Process for 2D inputs.
"""
struct BeliefModel
    gp
end

"""
Inputs:

    - X: a single point or an array of multiple points

Outputs:

    - μ, σ: a pair of expected value(s) and uncertainty(s) for the given point(s)
"""
function (beliefModel::BeliefModel)(x::Vector{<:Real})
    pred = only(marginals(beliefModel.gp([x])))
    return pred.μ, pred.σ
end

function (beliefModel::BeliefModel)(X::Vector{<:Vector{<:Real}})
    pred = marginals(beliefModel.gp(X))
    μ = getfield.(pred, :μ)
    σ = getfield.(pred, :σ)
    return μ, σ
end

"""
$SIGNATURES

Creates and returns a new belief model containing a GP. The GP is trained and
conditioned on the given samples.
"""
function generateBeliefModel(samples, region)
    # set up data
    X_train = getfield.(samples, :x)
    Y_train = getfield.(samples, :y)

    # set up hyperparameters
    a = (length(Y_train)>1 ? std(Y_train) : 0.5)/sqrt(2)
    b = (length(X_train)>1 ?
        mean(std(X_train)) :
        0.2*mean(mean([region.lb, region.ub])))
    θ0 = (;
          σ = positive(exp(a)),
          ℓ = positive(exp(b)),
          σn = positive(exp(0.001))
          )
    θ0_flat, unflatten = ParameterHandling.value_flatten(θ0)
    loss_flat = lossFunction(X_train, Y_train) ∘ unflatten

    # optimize hyperparameters (train)
    opt = optimize(loss_flat, θ0_flat, LBFGS())
    θ = unflatten(opt.minimizer)

    # produce optimized gp belief model
    f = GP(kernel(θ)) # prior gp
    fx = f(X_train, θ.σn^2+√eps())
    f_post = posterior(fx, Y_train) # gp conditioned on training samples
    beliefModel = BeliefModel(f_post)
    return beliefModel
end

"""
$SIGNATURES

A simple squared exponential kernel for the GP with parameters θ.
"""
kernel(θ) = θ.σ^2 * with_lengthscale(SqExponentialKernel(), θ.ℓ^2)

"""
$SIGNATURES

This function creates the loss function for training the GP. The negative log
marginal likelihood is used.
"""
function lossFunction(X, Y)
    # returns a function for the negative log marginal likelihood
    θ -> begin
        f = GP(kernel(θ))
        fx = f(X, θ.σn^2+√eps()) # eps to prevent numerical issues
        return -logpdf(fx, Y)
    end
end

end
