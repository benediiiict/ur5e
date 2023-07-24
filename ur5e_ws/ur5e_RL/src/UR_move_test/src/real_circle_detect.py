#!/usr/bin/python3

import sys
import rospy
import cv2 as cv
import numpy as np
import tf
import time
from geometry_msgs.msg import Transform
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from numpy.linalg import inv

import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
# Xfactor = 0.002541
# Yfactor = 0.0023456

# rospy.Subscriber('/ft_sensor_topic', Image, read_image) 

def coordinate_tf(cameraTFmatrix):
    rate = rospy.Rate(10.0)
    tfBuffer = tf2_ros.Buffer()
    tf_listener = TransformListener(tfBuffer)
    time.sleep(1.0)

    transTX = -0.2476571762
    transTY = 0.5351605723
    transTZ = 0.8882418124
    transRX = -0.9997695056
    transRY = 0.0153927116
    transRZ = 0.0007281098
    transRW = 0.0149489122

    try:
        # trans = tfBuffer.lookup_transform('base_link', 'camera_link', rospy.Time())
        # # print(trans.transform.translation)
        # # print(trans.transform.rotation)
        # base2camera = tf.TransformListener().fromTranslationRotation((trans.transform.translation.x,
        #                                                            trans.transform.translation.y,
        #                                                            trans.transform.translation.z), 
        #                                                            (trans.transform.rotation.x,
        #                                                             trans.transform.rotation.y,
        #                                                             trans.transform.rotation.z,
        #                                                             trans.transform.rotation.w))
        # print(base2camera)
        # _rtool_tool_trans = np.mat(np.identity(4))
        # obj_point = base2camera*camera_p
        # print(obj_point)


        base2camera = tf.TransformListener().fromTranslationRotation((transTX,
                                                                   transTY,
                                                                   transTZ), 
                                                                   (transRX,
                                                                    transRY,
                                                                    transRZ,
                                                                    transRW))
        print(base2camera)
        obj_point = base2camera*cameraTFmatrix
        print(obj_point)

        return obj_point
    

    
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rate.sleep()
        pass
    # base_trans_tool = TransformArray()
    # base_rot_tool = TransformArray()
    
    # try:
    #     (base_trans_tool, base_rot_tool) = tf.TransformListener().lookupTransform('/shoulder_link', '/upper_arm_link', rospy.Time.now())
    #     print(base_trans_tool)
    #     print(base_rot_tool)
    #     # return
    # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #     rospy.logwarn('lookupTransform for robot failed!, ' + "base_link" + ', ' + "tool0")
    #     # return 


def camera_parameter(u, v):
    # camera_i = np.array([
    #         [615.18, 0, 326],
    #         [0, 615.18, 237.415],
    #         [0, 0, 1]
    #         # [0.0031, 0, -0.993],
    #         # [0, 0.0031, -0.5554],
    #         # [0, 0, 1]
    # ])

    camera_i = np.matrix([
            [615.18, 0, 326],
            [0, 615.18, 237.415],
            [0, 0, 1]
    ])

    camera_i_inv = inv(camera_i)
    camera_uv = np.matrix([[u], [v], [1]])
    # camera_coordinate = camera_i_inv.dot(camera_uv)
    camera_coordinate = camera_i_inv*camera_uv
    # camera_coordinate = np.multiply(camera_i_inv,camera_uv)
    print(camera_coordinate)
    # camera_coordinate_add_dimen = np.array([camera_coordinate[0], camera_coordinate[1], 1, 1])
    camera_coordinate_4x4 = np.matrix([
                            [1, 0, 0, camera_coordinate[0]],
                            [0, 1, 0, camera_coordinate[1]],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]
    ])
    print(camera_coordinate_4x4)
    return camera_coordinate_4x4


def detect_circle(src):
    # gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # gray = cv.medianBlur(gray, 5)
    
    # rows = gray.shape[0]          # param1=100, param2=30,
    # circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
    #                            param1=100, param2=20,
    #                            minRadius=1, maxRadius=20)
    
    
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         center = (i[0], i[1])     # (x, y) coordinate in the screen 
    #         print(center)             # (306, 232)
    #         # circle center red
    #         cv.circle(src, center, 1, (0, 0, 255), 3)
    #         # circle outline
    #         radius = i[2]
    #         cv.circle(src, center, radius, (0, 255, 0), 3)


    camera_point = camera_parameter(300, 200) 
    # camera_point = camera_parameter(center[0], center[1])   
    print(camera_point)
    hole_point = coordinate_tf(camera_point)  
    print(hole_point)   


    # cv.imshow("detected circles", src)
    # cv.waitKey(0)

    return 0


def read_image():
    status = rospy.wait_for_message("/camera/color/image_raw", Image)      
    status = CvBridge().imgmsg_to_cv2(status, "bgr8")
    # print(status)
    print("===== OK!!! take hole picture =====")
    detect_circle(status)


if __name__ == "__main__":
    rospy.init_node("circle_detect")
    # main(sys.argv[1:])
    detect_circle()
    # read_image()



# def main(argv):
    
#     default_file = '/home/ben/work/ur5e_RL/src/UR_move_test/image_detect/image.png'
#     filename = argv[0] if len(argv) > 0 else default_file

#     # Loads an image
#     src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
#     # Check if image is loaded fine
#     if src is None:
#         print ('Error opening image!')
#         print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
#         return -1
    
    
#     gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    
    
#     gray = cv.medianBlur(gray, 5)
    
    
#     rows = gray.shape[0]
#     circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
#                                param1=100, param2=30,
#                                minRadius=1, maxRadius=20)
    
    
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for i in circles[0, :]:
#             center = (i[0], i[1])
#             # circle center
#             cv.circle(src, center, 1, (0, 0, 255), 3)
#             # circle outline
#             radius = i[2]
#             cv.circle(src, center, radius, (0, 255, 0), 3)
    
#     cv.imshow("detected circles", src)
#     cv.waitKey(0)

#     return 0


        # trans = np.mat(np.identity(4))
        # trans[2, :] = obj_point
        # print(np.array(obj_point[:3]).reshape(-1))
        # quat_result = tf.transformations.quaternion_from_matrix(trans)
        # print(quat_result)
        # # obj_point = np.multiply(base2camera,camera_p)