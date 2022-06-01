#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

bridge = CvBridge()

def rgb_callback(rgb_img):
    rgb_numpy_img = bridge.imgmsg_to_cv2(rgb_img, desired_encoding="passthrough")
    

def main():
    rospy.init_node("rgbd_subs", anonymous=True)
    rospy.Subscriber("/camera/color/image_raw/", Image, rgb_callback)

    rospy.spin()
if __name__ == "__main__":
    main()