# Code Parts and Usage Overview

The code is divided up into files to contain pieces with similar purposes or concepts in the algorithm. Each file has its own single module for defining a namespace used when importing its names into other files. Each module exports members intended for public access, but the code in this project explicitly names its imports to maintain clarity in what is used and where it comes from.

## Main.jl

The general scripting area and launching point for running the parts of the system. The main steps are initializing data, running the exploration algorithm, and visualizing results.

## Maps.jl

Types for holding and handling real and simulated data. Mainly used by [Missions.jl](@ref), but foundational for the other parts too. Its alias types `Location` and `SampleInput` are fundamental pieces for Samples and BeliefModels.

Main public types and functions:
- `Location`
- `SampleInput`
- `SampleOutput`
- `Map`
- `MultiMap`
- `imgToMap`
- `res`
- `pointToCell`
- `GroundTruth`

## Missions.jl

Functions for initializing mission data and the function for running the entire search mission. The entry-point to the actual adaptive sampling. This contains the main loop and most of the usage of Samples and BeliefModels.

Main public types and functions:
- `simMission`
- `ausMission`
- `rosMission`

## BeliefModels.jl

Everything to do with what is inferred about values in the environment. In practical terms: means, variances, and correlations. This is all built on Gaussian Processes.

Main public types and functions:
- `BeliefModel`
- `outputCorMat`

## Samples.jl

Everything to do with sampling values in the environment.

Main public types and functions:
- `Sample`
- `takeSamples`
- `selectSampleLocation`
- `SampleCost`

## SampleCosts.jl

Holds a variety of SampleCost functions used by [Samples.jl](@ref) in selecting a new sample location.

Main public types and functions:
- `SampleCost`
- `values`
- `BasicSampleCost`
- `NormedSampleCost`
- `MIPTSampleCost`
- `EIGFSampleCost`

## Paths.jl

Searching for paths on a 2D grid using A#. Its main use is to get the path cost (distance), but it can return the full path as well.

Main public types and functions:
- `PathCost`
- `finalOrientation`
- `getPath`

## Visualization.jl

Methods to visualize mission data. The form displayed is determined by what data types are passed in.

Main public types and functions:
- `visualize`

## ROSInterface.jl

The interface for passing data to and from other ROS nodes. It sets up an `adaptive_sampling` node and provides methods to handle the data. This is designed specifically for communication with Swagbot.

