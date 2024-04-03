# a script used to set up the project from cmake

using Pkg;

println("Instantiating Julia package")
Pkg.activate(".")
Pkg.instantiate()

println("Building PyCall")
ENV["PYTHON"] = "/usr/bin/python";
Pkg.build("PyCall")

println("Instantiating Julia app")
Pkg.activate("./app")
Pkg.instantiate()
