module Paths

using LinearAlgebra: norm
using DataStructures: PriorityQueue, dequeue!
using DocStringExtensions: SIGNATURES

using Environment: Map, res, pointToCell

"""
Struct for PathCost function data. Previous computations are kept track of in
its data. Can be used multiple times for the same start cell, saving
computation.

The costMap's cells contain the distance to them from the start cell. NaN is a
placeholder meaning no path has been calculated to that cell yet. Inf means
that cell is not reachable from the start cell.
"""
struct PathCost
    start
    costMap
    frontier
end

"""
$SIGNATURES

The constructor initializes the path search algorithm, created before each new
start cell.
"""
function PathCost(start_loc, occupancy)
    start = pointToCell(start_loc, occupancy)

    # initialize data structures with values for first cell
    costMap = Map(fill(NaN, size(occupancy)), occupancy.lb, occupancy.ub)
    costMap[occupancy] .= Inf
    costMap[start] = 0.0

    frontier = PriorityQueue{CartesianIndex{2}, Float64}()
    frontier[start] = 0.0

    PathCost(start, costMap, frontier)
end

"""
Search for a path to a desired goal location using the A* algorithm.

Returns the path cost, which will be Inf if it is unreachable.

Throws an error if no value can be determined.

If the cells of the path are desired, use the backpath function
(not yet implemented).
"""
function (S::PathCost)(x_goal)
    goal = pointToCell(x_goal, S.costMap)
    S.costMap[goal] |> !isnan && return S.costMap[goal]

    # update the frontier for the new goal
    for cell in keys(S.frontier)
        S.frontier[cell] = S.costMap[cell] + dist(cell, goal, res(S.costMap))
    end

    while !isempty(S.frontier)
        cell = dequeue!(S.frontier) # current cell to try
        cost = S.costMap[cell] # get the cost of this cell from the matrix

        # check if we've reached the goal
        cell == goal && return cost

        # now look all around the current cell
        for i in -1:1, j in -1:1
            i == j == 0 && continue

            new_cell = cell + CartesianIndex(i,j)
            checkbounds(Bool, S.costMap, new_cell) || continue

            new_cost = S.costMap[cell] + dist(cell, new_cell, res(S.costMap))

            if (S.costMap[new_cell] |> isnan ||
                new_cell ∈ keys(S.frontier) && S.costMap[new_cell] > new_cost)
                S.costMap[new_cell] = new_cost
                # actual cost so far plus heuristic
                S.frontier[new_cell] = new_cost + dist(new_cell, goal, res(S.costMap))
            end
        end
    end

    # this is likely caused by cells with valid values that are unconnected to
    # other accessible cells
    return Inf
end

dist(x1, x2, weights) = norm(Tuple(x2 - x1) .* weights)

"""
$SIGNATURES

Given a PathCost and a goal point, this function returns the angle of the
direction from penultimate cell to goal cell, effectively the direction at the
end of the path to the goal.
"""
function finalOrientation(S::PathCost, x_goal)
    # start with the ending cell
    goal = pointToCell(x_goal, S.costMap)
    S.costMap[goal] |> isnan && error("a path has not yet been searched for this cell")
    S.costMap[goal] |> isinf && error("this cell is unreachable")

    # get the direction from the neighbor with the minimum cost, ignore NaNs
    dif = argmin(CartesianIndex(i,j) for i in -1:1, j in -1:1 if !(i == j == 0)) do dif
        val = S.costMap[goal - dif]
        return (isnan(val) ? Inf : val)
    end

    # return the angle from the difference
    return atan(dif[2], dif[1])
end

end
