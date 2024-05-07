using AbstractGPs: Kernel, MOKernel, MOInput

struct CustomMOKernel{Tg,TA<:AbstractMatrix} <: MOKernel
    g::Tg
    A::TA
    function CustomMOKernel(g, A::AbstractMatrix)
        all(gi isa Kernel for gi in g) || error("`g` should be a collection of kernels")
        size(g) == size(A) ||
            error("Size of `A` not compatible with the given array of kernels `g`")
        return new{typeof(g),typeof(A)}(g, A)
    end
end

function (κ::CustomMOKernel)((x, px)::Tuple{Any,Int}, (y, py)::Tuple{Any,Int})
    cov_f = κ.A[px, py] * κ.g[px, py](x, y)
    return cov_f
end

# function kernelmatrix(k::CustomMOKernel, x::MOInput, y::MOInput)
#     x.out_dim == y.out_dim || error("`x` and `y` should have the same output dimension")
#     x.out_dim == size(k.A, 1) ||
#         error("Kernel not compatible with the given multi-output inputs")
#
#     # Weights matrix ((out_dim x out_dim) x length(k.g))
#     W = [col * col' for col in eachcol(k.A)]
#
#     # Latent kernel matrix ((N x N) x length(k.g))
#     H = [gi.(x.x, permutedims(y.x)) for gi in k.g]
#
#     # Weighted latent kernel matrix ((N*out_dim) x (N*out_dim))
#     W_H = kron(k.A, H)
#
#     return W_H
# end

function Base.show(io::IO, k::CustomMOKernel)
    return print(io, "Semi-parametric Latent Factor Multi-Output Kernel")
end

function Base.show(io::IO, ::MIME"text/plain", k::CustomMOKernel)
    print(io, "Semi-parametric Latent Factor Multi-Output Kernel\n\tgᵢ: ")
    return join(io, k.g, "\n\t\t")
end
