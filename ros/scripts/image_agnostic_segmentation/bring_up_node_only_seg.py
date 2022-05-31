#!/usr/bin/env python

from urllib import response
from image_agnostic_segmentation.ros.scripts.image_agnostic_segmentation.bring_up_node import handle_segment_image
import rospy
import sys
import numpy as np
import os
import open3d as o3d

import rospkg
import ros_numpy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose

from cv_bridge import CvBridge
bridge = CvBridge()

from image_agnostic_segmentaiton.srv import SegmentImage, SegmentImageResponse
from image_agnostic_segmentaiton.msg import ImagePixels
from sensor_msgs.msg import Image

rospack = rospkg.RosPack()
rospack.list()
pkg_path = rospack.get_path('image_agnostic_segmentation')
lib_path = os.path.join(pkg_path, '../scripts')
sys.path.insert(0, lib_path)
from agnostic_segmentation import agnostic_segmentation
from agnostic_segmentation import compute_grasp

def handle_segment_image(req):
    rospy.loginfo("Segmentation service called.")

    seg_img_pub = rospy.Publisher("segmented_image", Image, queue_size=10)
    response = SegmentImageResponse()

    rgb_image = bridge.imgmsg_to_cv2(req.rgb_image, desired_encoding='passthrough')
    c_matrix = np.array(req.cam_K_matrix).reshape((3,3))

    model_path = os.path.join(pkg_path, '../models/FAT_trained_Ml2R_bin_fine_tuned.pth')

    predictions = agnostic_segmentation.segment_image(rgb_image, model_path)
    seg_img = agnostic_segmentation.draw_segmented_image(rgb_image, predictions)
    seg_img_msg = bridge.cv2_to_imgmsg(seg_img, encoding="rgb8")
    seg_img_pub.publish(seg_img_msg)
    rospy.loginfo("Published segmented image.")

def main():
    rospy.init_node('image_agnostic_segmentation_rgb')
    s = rospy.Service('segment_image', SegmentImage, handle_segment_image)
    rospy.loginfo("Segmentation service started")
    rospy.spin()

if __name__ == "__main__":
    main()