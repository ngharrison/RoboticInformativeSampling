#!/usr/bin/env python
# should be set and run as an executable

# this file is just used for testing ROSInterface.jl

import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

def callback(pose, pub):
    rospy.sleep(1)
    pub.publish(True) # sortie finished

def main():
    rospy.init_node("sortie_sim")
    pub = rospy.Publisher("sortie_finished", Bool, queue_size=1)
    rospy.Subscriber("latest_sample", PoseStamped, callback, pub, queue_size=1)
    rospy.spin()

if __name__ == '__main__':
    main()
