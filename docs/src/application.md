# Application --- Parts and Usage Overview

Scripts that run the informative sampling code. This is the starting place for use of the repository and it uses the other parts: core and utils.

## missions

The place where code to run missions is located.

The launching point for running the informative sampling missions. The main steps are initializing data, running the algorithm, and visualizing/saving results.

The `example.jl` script can be run to see the full code in action.

## scripts

The place where code to analyze saved data is located.

## ros

A python files used to test simple usage with ROS.

Also contains the `server.jl` script which handles ROS service requests.

## maps

The place where maps used as data are stored.

## output

The place that files are saved from the code by default.
