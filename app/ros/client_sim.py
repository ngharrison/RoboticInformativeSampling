#!/usr/bin/env python

from __future__ import print_function

import sys
import rospy
from informative_sampling.srv import *
from informative_sampling.msg import Sample, Bounds

def generateBeliefModelClient(samples, bounds):
    rospy.wait_for_service('generate_belief_model')
    try:
        generateBeliefModel = rospy.ServiceProxy('generate_belief_model', GenerateBeliefModel)
        result = generateBeliefModel(samples, bounds)
        return result.params
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def createSamples():
    return [
        Sample([.1, .8], 1, 1.4),
        Sample([.5, .4], 1, .2),
        Sample([.2, .2], 2, .8),
        Sample([.9, .1], 2, .1),
    ], Bounds([0.0, 0.0], [1.0, 1.0])

if __name__ == "__main__":
    print("Requesting BeliefModel")
    response = generateBeliefModelClient(*createSamples())
    print("Received %s" % response)
