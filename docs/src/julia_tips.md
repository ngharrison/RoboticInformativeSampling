# Julia Tips

## Development Environment

These instructions focus on getting started up just using the REPL, which can be a sufficient set of tools. If you want to use an IDE, VSCode with the Julia extension is the main one recommended. It is supported by the Julia contributors and the community as a whole. You'll need to read their docs to learn how to use it.

## REPL

Julia ships with a useful REPL that combines an interpreter, a documentation browser, a package manager, and a shell. The four modes are accessed by typing the following keys:
- Interpreter: default mode
- Help: `?`
- Packages: `]`
- Shell: `;`
Type backspace or ctrl-c to exit a mode.

## Compilation

Julia code gets compiled the first time you run it. This means the first run will be slower and later ones much faster. This is also true when loading packages with `using` or `import`. So the typical way to run julia code is through a REPL that is kept open between runs in order to not re-compile. If a script is run directly from the command line using the julia interpreter, it will be re-compiled every time. Running code within an IDE will typically keep a REPL open for you.

Note: Julia 1.9 reduces the load and first-execution times considerably for modules that have not changed. Highly recommended.

## Revise.jl

This is a great package to use when developing code (not needed when only running it). Normally to update methods and variables that have changed, you have to manually re-run the changed code in the REPL. This package tracks modules that you have included with `using` or `import` and automatically updates the running environment with any changes. Simply run `using Revise` /before/ `using` anything else (i.e. running the Main.jl file) and all your changes within the project will be tracked.

## Unchangeable stuff

Julia doesn't allow changing type definitions. This means if you need to change what is within a `struct` or you need use the name of a function for something else, you will need to restart the REPL.

## Functions and Methods

In Julia lingo, a function is a type given to a family of methods, which are all called with the given function name. The methods are specific implementations or instantiations of that function, which are based on the number and types of the arguments passed in.

Example:
```
julia> length # the function
length (generic function with 226 methods)

julia> ?length("text") # one method
length(s::AbstractString) -> Int
...continued...

julia> ?length([1,2,3]) # a different method
length(A::AbstractArray)
...continued...
```

## Timing, Profiling, and Debugging

The following packages can be useful when developing and analyzing the code.

### Timing

The easiest way to time code is simply by putting `@time` in front of a line of code. If you want to time multiple lines, put them inside a `@time begin ... end` block.

To automate multiple runs of the code and get statistical information, use [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) and the `@benchmark` or `@btime` macros.

### Profiling

Reference: <https://docs.julialang.org/en/v1/manual/profile/>

`using Profile`

Same as timing, just put `@profile` before the code you want profiled. To see the output run `Profile.print()`.

I like using [StatProfilerHTML.jl](https://github.com/tkluck/StatProfilerHTML.jl) for viewing the output. Can also use [ProfileView.jl](https://github.com/timholy/ProfileView.jl). They each have their own shortcut commands for profiling and viewing. See their pages.

### Debugging

Reference: <https://github.com/JuliaDebug/Debugger.jl>

`using Debugger`

Put `@run` or `@enter` before code to debug. Use the commands from the docs in the REPL.

## Finding source module of object

To get the module that an object comes from, you can use
```julia
parentmodule(ImportedType) # will tell you which module a function or type comes from
parentmodule(typeof(var)) # for the object a variable contains, get the type first
```

In this project all names used are explicitly imported at the top of the file to help new developers.
