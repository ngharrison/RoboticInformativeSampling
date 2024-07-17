module BeliefModels

using LinearAlgebra: diag, PosDefException, norm, Diagonal
using AbstractGPs: GP, posterior, mean_and_var, mean_and_cov,
                   logpdf, cov, var, diag_Xt_invA_X, cholesky,
                   _symmetric, logdet, _sqmahal, Xt_invA_X
using IrrationalConstants: log2π
using Statistics: mean, std
using Optim: optimize, Options, NelderMead, LBFGS
using ParameterHandling: value_flatten, fixed, value
using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF

using ..Samples: SampleInput
using ..Kernels: multiMean, singleKernel, multiKernel, fullyConnectedCovNum,
                 slfmKernel, fullyConnectedCovMat, manyToOneCovNum, manyToOneCovMat,
                 initHyperparams, mtoKernel
using ..Maps: Bounds

export BeliefModel, outputCorMat, meanDerivAndVar

"""
$(TYPEDEF)

Abstract type. All subtypes of this will be callable with the same interface:
`X -> μ, σ` (SampleInputs -> means, standard deviations)
"""
abstract type BeliefModel end

"""
$(TYPEDEF)

Belief model struct and function for multiple outputs with 2D inputs.

Designed on top of a Multi-output Gaussian Process, but can still be used with a
single output.
"""
struct BeliefModelSimple{T} <: BeliefModel
    "posterior Gaussian Process used to do inference"
    gp
    "number of outputs of the GP"
    N
    "kernel function used in the GP"
    kernel::T
    "hyperparameters of the GP kernel function"
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
beliefModel = BeliefModel(samples, M.prior_samples, bounds)

# create a BeliefModelSimple
beliefModel = BeliefModel([M.prior_samples; samples], bounds)
```
"""
function BeliefModel(samples, prior_samples, bounds;
                     noise=(value=0.0, learned=false),
                     kernel=multiKernel)
    # create a simple belief model for the current samples
    current = BeliefModel(samples, bounds; noise, kernel)
    isempty(prior_samples) && return current
    # create a simple belief model for the prior current samples combined
    combined = BeliefModel([prior_samples; samples], bounds; noise, kernel)
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
function BeliefModel(samples, bounds::Bounds;
                     noise=(value=0.0, learned=false),
                     kernel=multiKernel)
    # set up training data
    X, Y_vals, Y_errs, N = extractSampleVals(samples)

    # choose noise
    σn = (noise.learned ? noise.value : fixed(noise.value))
    μ = fixed(calcMeans(X, Y_vals, N))
    θ0 = initHyperparams(X, Y_vals, bounds, N, kernel; μ, σn)

    # optimize hyperparameters (train)
    θ = optimizeLoss(createLossFunc(X, Y_vals, Y_errs, kernel), θ0)

    # produce optimized gp belief model
    fx = buildPriorGP(X, Y_errs, kernel, θ)
    f_post = posterior(fx, Y_vals) # gp conditioned on training samples

    return BeliefModelSimple(f_post, N, kernel, θ)
end

# Produce a belief model with pre-chosen hyperparams
function BeliefModel(samples, θ; kernel=multiKernel)
    X, Y_vals, Y_errs, N = extractSampleVals(samples)

    fx = buildPriorGP(X, Y_errs, kernel, θ)
    f_post = posterior(fx, Y_vals) # gp conditioned on training samples

    return BeliefModelSimple(f_post, N, kernel, θ)
end

function extractSampleVals(samples)
    X = getfield.(samples, :x)
    Y = getfield.(samples, :y)

    # split measurements if needed
    if Y isa AbstractArray{<:NTuple{2, <:Real}}
        Y_vals, Y_errs = first.(Y), last.(Y)
    else
        Y_vals, Y_errs = Y, 0.0
    end

    N = maximum(last, X) # number of outputs

    return X, Y_vals, Y_errs, N
end

function calcMeans(X, Y, N)
    mean_vals = zeros(N)
    nums = zeros(Int, N)
    for ((_, q), y) in zip(X, Y)
        mean_vals[q] += y
        nums[q] += 1
    end
    return mean_vals ./ nums
end

function buildPriorGP(X, Y_errs, kernel, θ, ϵ=0.0)
    f = GP(multiMean(θ), kernel(θ))
    σn = θ.σn isa AbstractArray ? (θ.σn[x[2]] for x in X) : θ.σn
    f(X, Y_errs.^2 .+ σn.^2 .+ √eps() .+ ϵ) # eps to prevent numerical issues
end

# the logpdf of a conditional distribution
function condlogpdf(fx, Y_vals)
    m, C_mat = mean_and_cov(fx)
    X = fx.x

    # partition
    i1 = filter(i->X[i][2]==1, eachindex(X))
    i2 = filter(i->X[i][2]!=1, eachindex(X))

    y1 = Y_vals[i1]
    y2 = Y_vals[i2]
    m1 = m[i1]
    m2 = m[i2]
    Σ11 = C_mat[i1,i1]
    Σ22 = cholesky(_symmetric(C_mat[i2,i2]))
    Σ21 = C_mat[i2,i1]

    # condition
    m_post = m1 + Σ21' * (Σ22 \ (y2 - m2))
    C_post = Σ11 - Xt_invA_X(Σ22, Σ21)

    return _logpdf(m_post, C_post, y1)
end

function _logpdf(m, C_mat, Y::AbstractVecOrMat{<:Real})
    C = cholesky(_symmetric(C_mat))
    T = promote_type(eltype(m), eltype(C), eltype(Y))
    return -((size(Y, 1) * T(log2π) + logdet(C)) .+ _sqmahal(m, C, Y)) ./ 2
end

"""
Inputs:
- `X`: a single sample input or an array of multiple
- `full_cov`: (optional) if this is true, returns the full covariance matrix
  in place of the vector of standard deviations

Outputs:
- `μ, σ`: a pair of expected value(s) and uncertainty(s) for the given point(s)

# Examples
```julia
X = [([.1, .2], 1),
     ([.2, .1], 2)]
μ, σ = beliefModel(X) # result: [μ1, μ2], [σ1, σ2]
```
"""
function (beliefModel::BeliefModelSimple)(x::SampleInput; kwargs...)
    return only.(beliefModel([x]); kwargs...)
end

function (beliefModel::BeliefModelSimple)(X::AbstractArray{SampleInput}; full_cov=false)
    func = full_cov ? mean_and_cov : mean_and_var
    μ, σ² = reshape.(func(beliefModel.gp, vec(X)), Ref(size(X)))
    return μ, .√clamp!(σ², 0.0, Inf) # avoid negative variances
end

function meanDerivAndVar(beliefModel::BeliefModelSimple, x::SampleInput)
    return only.(meanDerivAndVar(beliefModel, [x]))
end

function meanDerivAndVar(beliefModel::BeliefModelSimple, X::AbstractArray{SampleInput})
    Xv = vec(X)
    dims = size(X)
    f = beliefModel.gp
    C_xcond_x = cov(f.prior, f.data.x, Xv)
    m_deriv_norm = reshape(meanDerivNorm(Xv, f.data.x, C_xcond_x, beliefModel.θ.ℓ, f.data.α), dims)
    C_post_diag = reshape(var(f.prior, Xv) - diag_Xt_invA_X(f.data.C, C_xcond_x), dims)
    return m_deriv_norm, .√clamp!(C_post_diag, 0.0, Inf)
end

function meanDerivNorm(X, Xm, C_xcond_x, ℓ, α)
    m_deriv_dims = map(eachindex(X[1][1])) do i
        dif_mat = [xi[1][i] - xj[1][i] for xi in X, xj in Xm]
        k_prime = -dif_mat ./ ℓ^2
        (C_xcond_x' .* k_prime) * α
    end
    return sqrt.(sum(arr.^2 for arr in m_deriv_dims))
end

"""
Inputs:
- `X`: a single sample input or an array of multiple
- `full_cov`: (optional) if this is true, returns the full covariance matrix
  in place of the vector of standard deviations

Outputs:
- `μ, σ`: a pair of expected value(s) and uncertainty(s) for the given point(s)

# Examples
```julia
X = [([.1, .2], 1),
     ([.2, .1], 2)]
μ, σ = beliefModel(X) # result: [μ1, μ2], [σ1, σ2]
```
"""
function (beliefModel::BeliefModelSplit)(X::Union{SampleInput, AbstractArray{SampleInput}}; full_cov=false)
    μ, _ = beliefModel.combined(X; full_cov)
    _, σ = beliefModel.current(X; full_cov)
    return μ, σ
end

"""
$(TYPEDSIGNATURES)

Routine to optimize the lossFunc and return the optimal parameters `θ`.

Can pass in a different solver. NelderMead is picked as default for better speed
with about the same performance as LFBGS.
"""
function optimizeLoss(lossFunc, θ0; solver=NelderMead, iterations=5_000)
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
            @error e θ X Y_vals

            # NOTE this will probably break if reached with the multiKernel
            fx = buildPriorGP(X, Y_errs, kernel, θ, 1e-1*maximum(θ.σ))
            return -logpdf(fx, Y_vals)
        end
    end
end


"""
```julia
outputCorMat(beliefModel::BeliefModel)
```

Gives the correlation matrix between all outputs.
"""
function outputCorMat(bm::BeliefModelSimple{typeof(multiKernel)})
    σn = bm.θ.σn isa AbstractArray ? bm.θ.σn : fill(bm.θ.σn, bm.N)
    cov_mat = fullyConnectedCovMat(bm.θ.σ) .+ Diagonal(σn.^2)
    vars = diag(cov_mat)
    return @. cov_mat / √(vars * vars') # broadcast shorthand
end

function outputCorMat(bm::BeliefModelSimple{typeof(mtoKernel)})
    σn = bm.θ.σn isa AbstractArray ? bm.θ.σn : fill(bm.θ.σn, bm.N)
    cov_mat = manyToOneCovMat(bm.θ.σ) .+ Diagonal(σn.^2)
    vars = diag(cov_mat)
    return @. cov_mat / √(vars * vars') # broadcast shorthand
end

function outputCorMat(bm::BeliefModelSplit)
    return outputCorMat(bm.combined)
end

end
