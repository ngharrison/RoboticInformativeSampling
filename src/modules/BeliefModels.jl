module BeliefModels

using LinearAlgebra: I, diag
using AbstractGPs: GP, posterior, mean_and_var, mean_and_cov, logpdf, with_lengthscale,
                   SqExponentialKernel, IntrinsicCoregionMOKernel
using Statistics: mean, std
using Optim: optimize, Options, NelderMead, LBFGS
using ParameterHandling: value_flatten
using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF

using Samples: SampleInput

export BeliefModel, fullyConnectedCovMat, outputCorMat

include("SLFMKernel.jl")

"""
$(TYPEDEF)

Abstract type. All subtypes of this will be callable with the same interface:
`X -> μ, σ` (SampleInputs -> means, variances)
"""
abstract type BeliefModel end

"""
$(TYPEDEF)

Belief model struct and function for multiple outputs with 2D inputs.

Designed on top of a Multi-output Gaussian Process, but can still be used with a
single output.
"""
struct BeliefModelSimple <: BeliefModel
    gp
    θ
end

function Base.show(io::IO, bm::BeliefModelSimple)
    print(io, "BeliefModelSimple")
end
function Base.show(io::IO, ::MIME"text/plain", bm::BeliefModelSimple)
    print(io, "BeliefModelSimple:\n\tθ = $(bm.θ)")
end

"""
$(TYPEDEF)

A combination of two BeliefModelSimples. One is trained on current samples and
the other is trained on current and previous samples. The main purpose for this
is to use the current for mean estimates and the combined for variance
estimates.

This was used for a certain sample cost function and isn't used so much anymore.
Still could be a valuable idea.
"""
struct BeliefModelSplit <: BeliefModel
    current
    combined
end

function Base.show(io::IO, bm::BeliefModelSplit)
    print(io, "BeliefModelSplit")
end
function Base.show(io::IO, ::MIME"text/plain", bm::BeliefModelSplit)
    println(io, "BeliefModelSplit")
    println("\tcurrent::BeliefModelSimple")
    println("\tcombined::BeliefModelSimple")
end

"""
$(TYPEDSIGNATURES)

Creates and returns a new BeliefModel. A BeliefModelSimple is returned if there
are no prior samples, and it is trained and conditioned on the given samples.
Otherwise a BeliefModelSplit is returned, trained and conditioned on both the
samples and prior samples. Lower and upper bounds are used to initialize one of
the hyperparameters.

A noise standard deviation can optionally be passed in either as a single scalar
value for all samples or a vector of values, one for each sample.

# Examples
```julia
# create a BeliefModelSplit
beliefModel = BeliefModel(samples, M.prior_samples, lb, ub)

# create a BeliefModelSimple
beliefModel = BeliefModel([M.prior_samples; samples], lb, ub)
```
"""
function BeliefModel(samples, prior_samples, lb, ub; σn=1e-3, kernel=multiKernel)
    # create a simple belief model for the current samples
    current = BeliefModel(samples, lb, ub; σn, kernel)
    isempty(prior_samples) && return current
    # create a simple belief model for the prior current samples combined
    combined = BeliefModel([prior_samples; samples], lb, ub; σn, kernel)
    # split is a combination of the two
    return BeliefModelSplit(current, combined)
end

"""
$(TYPEDSIGNATURES)

Creates and returns a BeliefModelSimple with hyperparameters trained and
conditioned on the samples given.

A noise standard deviation can optionally be passed in either as a single scalar
value for all samples or a vector of values, one for each sample.
"""
function BeliefModel(samples, lb, ub; σn=1e-3, kernel=multiKernel)
    # set up training data
    X = getfield.(samples, :x)
    Y = getfield.(samples, :y)

    f_post, θ = buildOptimizedGP(X, Y, lb, ub; σn, kernel)
    return BeliefModelSimple(f_post, θ)
end

# no measurement noise
function buildOptimizedGP(X, Y_vals::AbstractArray{<:Real}, lb, ub; σn, kernel)
    Y_errs = 0.0

    θ0 = initHyperparams(X, Y_vals, lb, ub; σn)

    # optimize hyperparameters (train)
    θ = optimizeLoss(createLossFunc(X, Y_vals, Y_errs, kernel), θ0)

    # produce optimized gp belief model
    fx = buildPriorGP(X, Y_errs, kernel, θ)
    f_post = posterior(fx, Y_vals) # gp conditioned on training samples

    return f_post, θ
end

# with measurement noise
function buildOptimizedGP(X, Y::AbstractArray{<:NTuple{2, <:Real}}, lb, ub; σn, kernel)
    Y_vals, Y_errs = first.(Y), last.(Y)

    θ0 = initHyperparams(X, Y_vals, lb, ub) # no noise to learn

    # optimize hyperparameters (train)
    θ = optimizeLoss(createLossFunc(X, Y_vals, Y_errs, kernel), θ0)

    # produce optimized gp belief model
    fx = buildPriorGP(X, Y_errs, kernel, θ)
    f_post = posterior(fx, Y_vals) # gp conditioned on training samples

    return f_post, θ
end

function buildPriorGP(X, Y_errs, kernel, θ, ϵ=0.0)
    f = GP(kernel(θ))
    f(X, Y_errs.^2 .+ √eps() .+ ϵ) # eps to prevent numerical issues
end

"""
Inputs:
- `X`: a single sample input or an array of multiple
- `full_cov`: (optional) if this is true, returns the full covariance matrix
  in place of the vector of variances

Outputs:
- `μ, σ`: a pair of expected value(s) and uncertainty(s) for the given point(s)

# Examples
```julia
X = [([.1, .2], 1),
     ([.2, .1], 2)]
μ, σ = beliefModel(X) # result: [μ1, μ2], [σ1, σ2]
```
"""
function (beliefModel::BeliefModelSimple)(x::SampleInput; full_cov=false)
    func = full_cov ? mean_and_cov : mean_and_var
    μ, σ² = only.(func(beliefModel.gp, [x]))
    return μ, .√σ²
end

function (beliefModel::BeliefModelSimple)(X::Vector{SampleInput}; full_cov=false)
    func = full_cov ? mean_and_cov : mean_and_var
    μ, σ² = func(beliefModel.gp, X)
    return μ, .√σ²
end

"""
Inputs:
- `X`: a single sample input or an array of multiple
- `full_cov`: (optional) if this is true, returns the full covariance matrix
    in place of the vector of variances

Outputs:
- `μ, σ`: a pair of expected value(s) and uncertainty(s) for the given point(s)

# Examples
```julia
X = [([.1, .2], 1),
     ([.2, .1], 2)]
μ, σ = beliefModel(X) # result: [μ1, μ2], [σ1, σ2]
```
"""
function (beliefModel::BeliefModelSplit)(X::Union{SampleInput, Vector{SampleInput}}; full_cov=false)
    μ, _ = beliefModel.combined(X; full_cov)
    _, σ = beliefModel.current(X; full_cov)
    return μ, σ
end

"""
$(TYPEDSIGNATURES)

Creates the structure of hyperparameters for a MTGP and gives them initial values.
"""
function initHyperparams(X, Y_vals, lb, ub; kwargs...)
    T = maximum(last, X) # number of outputs
    n = fullyConnectedCovNum(T)
    # NOTE may change to all just 0.5
    # σ = (length(Y_vals)>1 ? std(Y_vals) : 0.5)/sqrt(2) * ones(n)
    σ = 0.5/sqrt(2) * ones(n)
    a = mean(ub .- lb)
    ℓ = length(X)==1 ? a : a/length(X) + mean(std(first.(X)))*(1-1/length(X))
    return (; σ, ℓ, kwargs...)
end

"""
$(TYPEDSIGNATURES)

Creates the structure of hyperparameters for a SLFM and gives them initial values.
"""
function initHyperparamsSLFM(X, Y_vals, lb, ub; kwargs...)
    T = maximum(last, X) # number of outputs
    # NOTE may change to all just 0.5
    # σ = (length(first.(Y_vals))>1 ? std(first.(Y_vals)) : 0.5)/sqrt(2) * ones(n)
    σ = 0.5/sqrt(2) * ones(T,T)
    a = mean(ub .- lb)
    ℓ = (length(X)==1 ? a : a/length(X) + mean(std(first.(X)))*(1-1/length(X))) * ones(T)
    return (; σ, ℓ, kwargs...)
end

"""
$(TYPEDSIGNATURES)

Routine to optimize the lossFunc and return the optimal parameters `θ`.

Can pass in a different solver. NelderMead is picked as default for better speed
with about the same performance as LFBGS.
"""
function optimizeLoss(lossFunc, θ0; solver=NelderMead, iterations=1_500)
    options = Options(; iterations)

    θ0_flat, unflatten = value_flatten(θ0)
    loss_flat = lossFunc ∘ unflatten

    opt = optimize(loss_flat, θ0_flat, solver(), options)
    @debug "model optimizer:" opt

    return unflatten(opt.minimizer)
end

"""
$(TYPEDSIGNATURES)

This function creates the loss function for training the GP. The negative log
marginal likelihood is used.
"""
function createLossFunc(X, Y_vals, Y_errs, kernel)
    # returns a function for the negative log marginal likelihood
    θ -> begin
        try
            fx = buildPriorGP(X, Y_errs, kernel, θ)
            return -logpdf(fx, Y_vals)
        catch e
            # for PosDefException
            # this seems to happen when θ.σ is extremely large and θ.ℓ is
            # much bigger than the search region dimensions
            @error e θ X Y_vals, Y_errs

            # NOTE this will probably break if reached with the multiKernel
            fx = buildPriorGP(X, Y_errs, kernel, θ, 1e-1*θ.σ)
            return -logpdf(fx, Y_vals)
        end
    end
end


## Kernel stuff

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

"""
```julia
outputCorMat(beliefModel::BeliefModel)
```

Gives the correlation matrix between all outputs.
"""
function outputCorMat(beliefModel::BeliefModelSimple)
    cov_mat = fullyConnectedCovMat(beliefModel.θ.σ)
    vars = diag(cov_mat)
    return @. cov_mat / √(vars * vars') # broadcast shorthand
end

function outputCorMat(beliefModel::BeliefModelSplit)
    return outputCorMat(beliefModel.combined)
end

end
