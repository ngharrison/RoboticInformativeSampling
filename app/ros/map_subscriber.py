#!/usr/bin/env python
# should be set and run as an executable

# this file is just used for testing ROSInterface.jl

import rospy
from std_msgs.msg import Float64MultiArray

def callback(data):
    print("Data received!")
    print(data)

def main():
    rospy.init_node("map_subscriber")
    rospy.Subscriber("/informative_sampling/pred_array_2",
                     Float64MultiArray, callback, queue_size=1)
    rospy.spin()

if __name__ == '__main__':
    main()
