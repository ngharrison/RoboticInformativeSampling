# Utilities --- Parts and Usage Overview

Modules to help run the informative sampling.

### DataIO.jl

Handling data in and out.

Main public types and functions:
- `normalize`
- `spatialAve`
- `imgToMap`
- `save`
- `maps_dir`
- `output_dir`
- `output_ext`
- `saveBeliefMapToPng`

```@autodocs
Modules = [DataIO]
```

### Visualization.jl

Methods to visualize mission data. The form displayed is determined by what data types are passed in.

Main public types and functions:
- `visualize`

```@autodocs
Modules = [Visualization]
```

### Metrics.jl

A function to calculate the metrics from a mission and belief model.

Main public types and functions:
- `calcMetrics`

```@autodocs
Modules = [Metrics]
```
