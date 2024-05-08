#!/usr/bin/env julia

# a script used to set up the project

using Pkg;

println("Instantiating package core")
Pkg.activate(Base.source_dir() * "/core")
Pkg.instantiate()

println("Building PyCall")
ENV["PYTHON"] = "/usr/bin/python";
Pkg.build("PyCall")

println("Instantiating package utils")
Pkg.activate(Base.source_dir() * "/utils")
Pkg.instantiate()

println("Instantiating package app")
Pkg.activate(Base.source_dir() * "/app")
Pkg.instantiate()

println("Instantiating package docs")
Pkg.activate(Base.source_dir() * "/docs")
Pkg.instantiate()
