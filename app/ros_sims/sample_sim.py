#!/usr/bin/env python
# should be set and run as an executable

# this file is just used for testing ROSInterface.jl

import rospy
from std_msgs.msg import Float32
from random import random

def loop(pubs):
    loop_rate = rospy.Rate(2.0)
    while not rospy.is_shutdown():
        for pub in pubs:
            pub.publish(Float32(random()))
        loop_rate.sleep()

def main():
    rospy.init_node("sample_sim")
    pubs = [rospy.Publisher("value1", Float32, queue_size=1),
            rospy.Publisher("value2", Float32, queue_size=1)]
    loop(pubs)

if __name__ == '__main__':
    main()
