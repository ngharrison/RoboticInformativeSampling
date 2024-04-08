# Package Parts and Usage Overview

The code is divided up into files to contain pieces with similar purposes or concepts in the algorithm. Each file has its own single module for defining a namespace used when importing its names into other files. Each module exports members intended for public access, but the code in this project explicitly names its imports to maintain clarity in what is used and where it comes from.

## Maps.jl

Types for holding and handling real and simulated data. Mainly used by [Missions.jl](@ref), but foundational for the other parts too. Its alias types `Location` and `SampleInput` are fundamental pieces for Samples and BeliefModels.

Main public types and functions:
- `Location`
- `SampleInput`
- `SampleOutput`
- `Map`
- `imgToMap`
- `res`
- `pointToCell`
- `GroundTruth`

```@autodocs
Modules = [Maps]
```

## Missions.jl

Functions for initializing mission data and the function for running the entire search mission. The entry-point to the actual adaptive sampling. This contains the main loop and most of the usage of Samples and BeliefModels.

Main public types and functions:
- `simMission`
- `ausMission`
- `rosMission`

```@autodocs
Modules = [Missions]
```

## BeliefModels.jl

Everything to do with what is inferred about values in the environment. In practical terms: means, variances, and correlations. This is all built on Gaussian Processes.

Main public types and functions:
- `BeliefModel`
- `outputCorMat`

```@autodocs
Modules = [BeliefModels]
```

## Samples.jl

Everything to do with sampling values in the environment.

Main public types and functions:
- `Sample`
- `MapsSampler`
- `takeSamples`
- `selectSampleLocation`
- `SampleCost`

```@autodocs
Modules = [Samples]
```

## SampleCosts.jl

Holds a variety of SampleCost functions used by [Samples.jl](@ref) in selecting a new sample location.

Main public types and functions:
- `SampleCost`
- `values`
- `BasicSampleCost`
- `NormedSampleCost`
- `MIPTSampleCost`
- `EIGFSampleCost`

```@autodocs
Modules = [SampleCosts]
```

## Paths.jl

Searching for paths on a 2D grid using A*. Its main use is to get the path cost (distance), but it can return the full path as well.

Main public types and functions:
- `PathCost`
- `finalOrientation`
- `getPath`

```@autodocs
Modules = [Paths]
```

## ROSInterface.jl

The interface for passing data to and from other ROS nodes. It sets up an `adaptive_sampling` node and provides methods to handle the data. This is designed specifically for communication with Swagbot.

Main public types and functions:
- `ROSConnection`

```@autodocs
Modules = [ROSInterface]
```
