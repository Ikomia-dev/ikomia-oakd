from pathlib import Path
import depthai as dai
import cv2
import sys

# Importing from parent folder
sys.path.insert(0, str(Path(__file__).parent.parent)) # move to parent path
from utils.OakSingleModelRunner import OakSingleModelRunner
from utils.draw import displayFPS, drawROI


# Set few parameters
nn_path = str(Path(__file__).parent) + "/../models/coronamask.blob"
labels = ["background", "no mask", "mask", "no mask"]


# Process function
def process(runner):
    left_frame = runner.left_cam_output_queue.get().getCvFrame()
    detections = runner.nn_output_queue.get().detections
    for detection in detections:
        drawROI(left_frame, (detection.xmin,detection.ymin), (detection.xmax,detection.ymax), label=runner.labels[detection.label], confidence=detection.confidence)
    
    right_frame = runner.right_cam_output_queue.get().getCvFrame()
    detections = runner.nn_output_queue.get().detections
    for detection in detections:
        drawROI(right_frame, (detection.xmin,detection.ymin), (detection.xmax,detection.ymax), label=runner.labels[detection.label], confidence=detection.confidence)
    
    displayFPS(left_frame, runner.getFPS())
    displayFPS(right_frame, runner.getFPS())
    cv2.imshow("left", left_frame)
    cv2.imshow("right", right_frame)
    
    


runner = OakSingleModelRunner() 

# Configure left and right cameras
runner.setLeftCamera(300, 300, sensor_resolution=dai.MonoCameraProperties.SensorResolution.THE_400_P)
runner.setRightCamera(300, 300, sensor_resolution=dai.MonoCameraProperties.SensorResolution.THE_400_P)
runner.left_cam_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
runner.right_cam_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

# runner.setRightCamera(sensor_resolution=dai.MonoCameraProperties.SensorResolution.THE_720_P)


# Configure stereo depth
# runner.setDepth(sensor_resolution=dai.MonoCameraProperties.SensorResolution.THE_400_P)
# runner.stereo.setOutputDepth(True)
# runner.stereo.setConfidenceThreshold(255)

# Configure neural network model and init input / output streams
runner.setMobileNetDetectionModel(path=nn_path, treshold=0.3, link_middle_cam=False)
runner.right_cam_manip.out.link(runner.nn.input)
runner.left_cam_manip.out.link(runner.nn.input)

# runner.nn.setDepthLowerThreshold(250)
# runner.nn.setDepthUpperThreshold(5000)
runner.labels = labels

# Start processing loop
runner.run(process=process)