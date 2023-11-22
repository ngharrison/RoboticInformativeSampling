# Code Patterns

## Naming

The chosen naming conventions for the project:
- Types (data structures) are `UpperCamelCase`
- Functions (and functors) are `lowerCamelCase`
- Simple constants are `CAPITAL_SNAKE_CASE`
- All other variables are `snake_case`

## Data structures and functions

A consistent pattern is followed for most data and functions in this repo using the Julia language feature of functors. A functor is a combination of a struct and a function. In many places in the code, you will find the following in order:
1. a struct definition -- what data the functor contains
2. zero to many constructor definitions -- how it is initialized
3. zero to many method definitions -- what the functor does when it is called (possibly with arguments)


As an example, here is a simplified version of the Map functor:
```julia
# struct definition
struct Map
    data
    lb
    ub
end

# constructor definition
Map(data) = Map(data, [0.0, 0.0], [1.0, 1.0])

# method definition
function (map::Map)(x::Location)
    checkBounds(x, map)
    map[pointToCell(x, map)]
end
```

A functor's methods are meant to be the primary purpose of that object. e.g. a BeliefModel returns the belief, a SampleCost returns the sample cost, etc. If an object's data are used in any secondary way, it will be its own separate function with its own name, and the object will be passed in directly.
