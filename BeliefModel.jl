module BeliefModel

using AbstractGPs
using StatsFuns
using Optim
using ParameterHandling

export generateBeliefModel, getBelief, lossFunction

function generateBeliefModel(region, samples)
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
    return f_post
end

function getBelief(x::Vector{Float64}, belief_model)
    # returns the expected value and uncertainty for a single point
    pred = only(marginals(belief_model([x])))
    return pred.μ, pred.σ
end

function getBelief(X::Vector{Vector{Float64}}, belief_model)
    # returns the expected value and uncertainty for multiple points
    pred = marginals(belief_model(X))
    μ = getfield.(pred, :μ)
    σ = getfield.(pred, :σ)
    return μ, σ
end

kernel(θ) = θ.σ^2 * with_lengthscale(SqExponentialKernel(), θ.ℓ^2)

function lossFunction(X, Y)
    # returns a function for the negative log marginal likelihood
    θ -> begin
        f = GP(kernel(θ))
        fx = f(X, θ.σn^2+√eps()) # eps to prevent numerical issues
        return -logpdf(fx, Y)
    end
end

end
