
# helper methods
function normalize(a)
    l, h = extrema(filter(!isnan, a))
    return (a .- l) ./ (h - l)
end

function spatialAve(M, extent=1)
    N = zero(M)
    for i in axes(M,1), j in axes(M,2)
        tot = 0
        count = 0
        for k in -extent:extent, l in -extent:extent
            m = i + k
            n = j + l
            if 1 <= m <= size(M,1) && 1 <= n <= size(M,2) && !isnan(M[m,n])
                tot += M[m,n]
                count += 1
            end
        end
        N[i,j] = tot/count
    end
    return N
end
