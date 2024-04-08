# Application Parts and Usage Overview

## missions

The place where code to run missions is located.

The launching point for running the informative sampling missions. The main steps are initializing data, running the algorithm, and visualizing/saving results.

The `example.jl` script can be run to see the full code in action.

## scripts

The place where code to analyze saved data is located.

## utils

Modules to help run the informative sampling.

### DataIO.jl

Handling data in and out.

Exports:
- `normalize`
- `spatialAve`
- `imgToMap`
- `save`
- `maps_dir`
- `output_dir`
- `output_ext`
- `saveBeliefMapToPng`

### Visualization.jl

Methods to visualize mission data. The form displayed is determined by what data types are passed in.

Main public types and functions:
- `visualize`

### Metrics.jl

A function to calculate the metrics from a mission and belief model.

Exports:
- `calcMetrics`
