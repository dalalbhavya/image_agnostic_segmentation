#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import sys
import rospkg
import ros_numpy
import time
import numpy as np

rospack = rospkg.RosPack()
rospack.list()
pkg_path = rospack.get_path("image_agnostic_segmentation")
lib_path = os.path.join(pkg_path, "../scripts")
sys.path.insert(0, lib_path)

from agnostic_segmentation import agnostic_segmentation

bridge = CvBridge()
time_taken = list()

def rgb_callback(rgb_img):
    pub = rospy.Publisher("/seg_img", Image, queue_size=10)
    rate = rospy.Rate(10)

    #convert image from sensor_msgs/Image to Numpy array
    SCALE_FACTOR = 0.5
    rgb_numpy_img = bridge.imgmsg_to_cv2(rgb_img, desired_encoding="passthrough")
    rgb_numpy_img = cv2.resize(rgb_numpy_img, (int(rgb_numpy_img.shape[1]*SCALE_FACTOR), int(rgb_numpy_img.shape[0]*SCALE_FACTOR)))

    #Segment Image
    model_path = model_path = os.path.join(pkg_path, '../models/FAT_trained_Ml2R_bin_fine_tuned.pth')
    start_time = time.time()
    predictions = agnostic_segmentation.segment_image(rgb_numpy_img, model_path)
    time_taken.append(time.time() - start_time)
    seg_rgb_img = agnostic_segmentation.draw_segmented_image(rgb_numpy_img, predictions)
    seg_rgb_img_msg = bridge.cv2_to_imgmsg(seg_rgb_img, encoding="rgb8")
    
    pub.publish(seg_rgb_img_msg)
    rospy.loginfo("Mean Time taken: " + str(np.mean(time_taken)))
    

def main():
    rospy.init_node("rgbd_subs", anonymous=True)
    rospy.Subscriber("/camera/color/image_raw/", Image, rgb_callback)

    rospy.spin()
if __name__ == "__main__":
    main()