
# this script must be run from within the InformativeSampling directory for file
# paths to work

from julia import Pkg, Main

Pkg.activate('app')

import julia.MultiQuantityGPs
import julia.GridMaps

from julia.InformativeSampling import Samples, SampleCosts, Missions

# from julia.InformativeSamplingUtils import DataIO, Visualization

# from julia.LinearAlgebra import I, norm
# from julia.Statistics import mean, cor
# from julia.Random import seed_b

# test creating simple belief model
samples = [
    MultiQuantityGPs.MQSample((([.1, .8], 1), 1.4)),
    MultiQuantityGPs.MQSample((([.5, .4], 1), .2)),
    MultiQuantityGPs.MQSample((([.2, .2], 2), .8)),
    MultiQuantityGPs.MQSample((([.9, .1], 2), .1)),
]

bounds = Main.eval('(lower=[0.0, 0.0], upper=[1.0, 1.0])')

noise_value = 0.0
noise_learn = False

beliefModel = MultiQuantityGPs.MQGP(samples, bounds=bounds, noise_value=noise_value, noise_learn=noise_learn)

print(beliefModel.θ)

#* test running full julia script

Main.include('app/missions/example.jl')

print(Main.samples)

print(Main.beliefs[-1].θ)
