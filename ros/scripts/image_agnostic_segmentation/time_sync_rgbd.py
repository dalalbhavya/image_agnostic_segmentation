#!/usr/bin/env python
from locale import CHAR_MAX
import numpy as np
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseArray
import message_filters
from cv_bridge import CvBridge
import cv2
import os
import sys
import rospkg
import time

bridge = CvBridge()
rospack = rospkg.RosPack()
rospack.list()
pkg_path = rospack.get_path("image_agnostic_segmentation")
lib_path = os.path.join(pkg_path, '../scripts')
sys.path.insert(0, lib_path)

from agnostic_segmentation import agnostic_segmentation
from agnostic_segmentation import compute_grasp

def rgbd_callback(rgb_img_msg, depth_img_msg):
    rospy.loginfo("Time synchronized RGBD image received")
    start_time = time.time()
    seg_img_pub = rospy.Publisher("segmented_image", Image, queue_size=10)
    grasp_img_pub = rospy.Publisher("grasp_image", Image, queue_size=10)
    grasp_poses_pub = rospy.Publisher("grasp_poses", PoseArray, queue_size=10)

    rgb_img = bridge.imgmsg_to_cv2(rgb_img_msg, desired_encoding="passthrough")
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    depth_img = bridge.imgmsg_to_cv2(depth_img_msg, desired_encoding="passthrough")
    camera_K_matrix = np.array([610.822998046875, 0.0, 432.04693603515625, 0.0, 609.3736572265625, 233.91339111328125, 0.0, 0.0, 1.0]).reshape((3,3))
    
    model_path = os.path.join(pkg_path, '../models/FAT_trained_Ml2R_bin_fine_tuned.pth')

    predictions = agnostic_segmentation.segment_image(rgb_img, model_path)
    seg_img = agnostic_segmentation.draw_segmented_image(rgb_img, predictions)
    seg_img_msg = bridge.cv2_to_imgmsg(seg_img, encoding="rgb8")
    seg_img_pub.publish(seg_img_msg)
    rospy.loginfo("Published segmented image.")
    rospy.loginfo(str(rgb_img.shape))

    #print("Predictions Keys:", predictions["instances"].to("cpu"))

    objects_clouds = compute_grasp.make_predicted_objects_clouds(rgb_img, depth_img, camera_K_matrix, predictions)
    print(objects_clouds)
    suction_pts = compute_grasp.compute_suction_points(predictions, objects_clouds)
    suction_pts_image = compute_grasp.visualize_suction_points(seg_img, camera_K_matrix, suction_pts)
    suction_pts_image_msg = bridge.cv2_to_imgmsg(suction_pts_image, encoding="rgb8")
    grasp_img_pub.publish(suction_pts_image_msg)
    rospy.loginfo("Time taken: " + str(time.time() - start_time))



    
def main():
    rospy.init_node("time_sync_rgbd", anonymous=True)
    rgb_img_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
    depth_img_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
    time_sync = message_filters.TimeSynchronizer([rgb_img_sub, depth_img_sub], 10)
    time_sync.registerCallback(rgbd_callback)

    rospy.spin()

if __name__ == "__main__":
    
    main()