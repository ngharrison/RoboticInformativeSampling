#!/usr/bin/env julia

# a script used to set up the project

using Pkg;

println("Instantiating package core")
Pkg.activate(Base.source_dir() * "/core")
Pkg.instantiate()

println("Building PyCall")
ros_version = readchomp(`rosversion -d`)
println("ROS version is ", ros_version)
python_exe = "/usr/bin/python" * (ros_version == "noetic" ? "3" : "")
println("Python executable is ", python_exe)
ENV["PYTHON"] = python_exe
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
