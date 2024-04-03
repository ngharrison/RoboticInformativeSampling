#!/usr/bin/env julia

# a script used to set up the project

using Pkg;

println("Instantiating Julia package")
Pkg.activate(Base.source_dir())
Pkg.instantiate()

println("Building PyCall")
ENV["PYTHON"] = "/usr/bin/python";
Pkg.build("PyCall")

println("Instantiating Julia app")
Pkg.activate(Base.source_dir() * "/app")
Pkg.instantiate()
