import depthai as dai


import cv2
import numpy as np


def updateSpatialCalculatorConfig(spatial_calculator_config, point_min, point_max, lower_threshold=250, upper_threshold=5000):
    spatial_config_data = dai.SpatialLocationCalculatorConfigData()
    spatial_config_data.depthThresholds.lowerThreshold = lower_threshold
    spatial_config_data.depthThresholds.upperThreshold = upper_threshold
    spatial_config_data.roi = dai.Rect(point_min, point_max)
    spatial_calculator_config.addROI(spatial_config_data)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
        return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]