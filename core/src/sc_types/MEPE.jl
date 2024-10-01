
## This isn't finished, don't work

"""
$(TYPEDEF)
"""
struct MEPE <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function MEPE(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    belief_max = nothing

    beta_hat = compute_beta_hat(obj, obj.R)
    H = obj.F * ((obj.F' * obj.F) \ obj.F')
    d = obj.Y - obj.F * beta_hat
    inv_R = inv(obj.R)
    e_CV_fast = zeros(obj.m, 1)
    for i = 1:obj.m

        e_CV_fast(i) = ((inv_R(i, :) * (d + H(:, i) * (d(i) / (1 - H(i, i))))) / (inv_R(i, i)))^2
    end

    MEPE(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::MEPE, loc)
    return (0.0, 0.0, 0.0, 0.0)
end

function (sc::MEPE)(loc)

    # get diff from previous model pred
    if length(sc.samples) == 1
        e_true = Inf
    else
        e_true = abs(sc.prev_belief(sc.samples[end].x) - sc.samples[end].y)
    end

    # get LOOCV approximation

    _, σ = sc.beliefModel((loc, 1)) # mean and standard deviation

    d = minimum(norm(sample.x[1] - loc) for sample in sc.samples)

    # set alpha
    if length(sc.samples) == 1
        α = 0.5
    else
        α = 0.99*minimum(0.5*(e_true^2 / e_CV), 1)
    end

    return sc.occupancy(loc) ? Inf : α*vals[1] + (1 - α)*σ^2
end

function alpha(err)

end

function errLOOCV(loc)

    return err
end
