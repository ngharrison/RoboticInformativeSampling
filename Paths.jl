module Paths

using LinearAlgebra
using DataStructures
using Environment

export PathCost

# this is used to initialize the algorithm and keep track of previous computations
struct PathCost
    start
    costMap
    frontier
end

function PathCost(x_start, obsMap)
    start = pointToIndex(x_start, obsMap)

    # initialize data structures with values for first cell
    costMap = Map(fill(NaN, size(obsMap)), obsMap.res)
    costMap[obsMap] .= Inf
    costMap[start] = 0.0

    frontier = PriorityQueue{CartesianIndex{2}, Float64}()
    frontier[start] = 0.0

    PathCost(start, costMap, frontier)
end

# search for a desired goal cell using the A* algorithm
# returns the path cost
function (S::PathCost)(x_goal)
    goal = pointToIndex(x_goal, S.costMap)
    S.costMap[goal] |> !isnan && return S.costMap[goal]

    # update the frontier for the new goal
    for cell in keys(S.frontier)
        S.frontier[cell] = S.costMap[cell] + norm(Tuple(goal - cell))
    end

    while !isempty(S.frontier)
        cell = dequeue!(S.frontier) # current cell to try
        cost = S.costMap[cell] # get the cost of this cell from the matrix

        # check if we've reached the goal
        cell == goal && return cost # we're done

        # now look all around the current cell
        for i in -1:1, j in -1:1
            i == j == 0 && continue

            new_cell = cell + CartesianIndex(i,j)
            checkbounds(Bool, S.costMap, new_cell) || continue

            new_cost = S.costMap[cell] + norm(Tuple(new_cell - cell) .* S.costMap.res)

            if (S.costMap[new_cell] |> isnan ||
                new_cell ∈ keys(S.frontier) && S.costMap[new_cell] > new_cost)
                S.costMap[new_cell] = new_cost
                # actual cost so far plus heuristic
                S.frontier[new_cell] = new_cost + norm(Tuple(goal - new_cell))
            end
        end
    end

    error("no path found")
end

end
