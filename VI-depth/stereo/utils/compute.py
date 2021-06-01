import depthai as dai
import numpy as np
import math
import cv2



def updateSpatialCalculatorConfig(spatial_calculator_config, point_min, point_max, lower_threshold=250, upper_threshold=5000):
    spatial_config_data = dai.SpatialLocationCalculatorConfigData()
    spatial_config_data.depthThresholds.lowerThreshold = lower_threshold
    spatial_config_data.depthThresholds.upperThreshold = upper_threshold
    spatial_config_data.roi = dai.Rect(point_min, point_max)
    spatial_calculator_config.addROI(spatial_config_data)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


def get_landmark_3d(landmark, focal_length=842, size=640):
    landmark_norm = 0.5 - np.array(landmark)
    landmark_image_coord = landmark_norm * size

    landmark_spherical_coord = [math.atan2(landmark_image_coord[0], focal_length),
                                -math.atan2(landmark_image_coord[1], focal_length) + math.pi / 2]

    landmarks_3D = [
        math.sin(landmark_spherical_coord[1]) * math.cos(landmark_spherical_coord[0]),
        math.sin(landmark_spherical_coord[1]) * math.sin(landmark_spherical_coord[0]),
        math.cos(landmark_spherical_coord[1])
    ]

    return landmarks_3D


def get_vector_intersection(left_vector, left_camera_position, right_vector, right_camera_position):
    if(len(left_vector)<2 or len(right_vector)<2):
        return []
    n = np.cross(left_vector, right_vector)
    n1 = np.cross(left_vector, n)
    n2 = np.cross(right_vector, n)

    top = np.dot(np.subtract(right_camera_position, left_camera_position), n2)
    bottom = np.dot(left_vector, n2)
    divided = top / bottom
    mult = divided * left_vector
    c1 = left_camera_position + mult

    top = np.dot(np.subtract(left_camera_position, right_camera_position), n1)
    bottom = np.dot(right_vector, n1)
    divided = top / bottom
    mult = divided * right_vector
    c2 = right_camera_position + mult

    center = (c1 + c2) / 2
    return center