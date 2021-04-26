import depthai as dai


def updateSpatialCalculatorConfig(spatial_calculator_config, point_min, point_max, lower_threshold=250, upper_threshold=5000):
    spatial_config_data = dai.SpatialLocationCalculatorConfigData()
    spatial_config_data.depthThresholds.lowerThreshold = lower_threshold
    spatial_config_data.depthThresholds.upperThreshold = upper_threshold
    spatial_config_data.roi = dai.Rect(point_min, point_max)
    spatial_calculator_config.addROI(spatial_config_data)