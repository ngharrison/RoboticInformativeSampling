using AbstractGPs: Kernel, MOKernel, MOInput, LatentFactorMOKernel

@doc raw"""
    SLFMMOKernel(g::AbstractVector{<:Kernel}, A::AbstractMatrix)

Kernel associated with the semiparametric latent factor model.

# Definition

For inputs ``x, x'`` and output dimensions ``p, p''``, the kernel is defined as[^STJ]
```math
k\big((x, p), (x', p')\big) = \sum^{Q}_{q=1} A_{p q}g_q(x, x')A_{p' q},
```
where ``g_1, \ldots, g_Q`` are ``Q`` kernels, one for each latent process, and ``A`` is a matrix of weights for the kernels of
size ``m \times Q``.

[^STJ]: M. Seeger, Y. Teh, & M. I. Jordan (2005). [Semiparametric Latent Factor Models](https://infoscience.epfl.ch/record/161465/files/slfm-long.pdf).
"""
struct SLFMMOKernel{Tg,TA<:AbstractMatrix} <: MOKernel
    g::Tg
    A::TA
    function SLFMMOKernel(g, A::AbstractMatrix)
        all(gi isa Kernel for gi in g) || error("`g` should be an collection of kernels")
        length(g) == size(A, 2) ||
            error("Size of `A` not compatible with the given array of kernels `g`")
        return new{typeof(g),typeof(A)}(g, A)
    end
end

function (κ::SLFMMOKernel)((x, px)::Tuple{Any,Int}, (y, py)::Tuple{Any,Int})
    cov_f = sum(κ.A[px, q] * κ.g[q](x, y) * κ.A[py, q] for q in 1:length(κ.g))
    return cov_f
end

function kernelmatrix(k::SLFMMOKernel, x::MOInput, y::MOInput)
    x.out_dim == y.out_dim || error("`x` and `y` should have the same output dimension")
    x.out_dim == size(k.A, 1) ||
        error("Kernel not compatible with the given multi-output inputs")

    # Weights matrix ((out_dim x out_dim) x length(k.g))
    W = [col * col' for col in eachcol(k.A)]

    # Latent kernel matrix ((N x N) x length(k.g))
    H = [gi.(x.x, permutedims(y.x)) for gi in k.g]

    # Weighted latent kernel matrix ((N*out_dim) x (N*out_dim))
    W_H = sum(kron(Wi, Hi) for (Wi, Hi) in zip(W, H))

    return W_H
end

function Base.show(io::IO, k::SLFMMOKernel)
    return print(io, "Semi-parametric Latent Factor Multi-Output Kernel")
end

function Base.show(io::IO, ::MIME"text/plain", k::SLFMMOKernel)
    print(io, "Semi-parametric Latent Factor Multi-Output Kernel\n\tgᵢ: ")
    return join(io, k.g, "\n\t\t")
end
