from pathlib import Path
import depthai as dai
import cv2
import sys

# Importing from parent folder
sys.path.insert(0, str(Path(__file__).parent.parent)) # move to parent path
from utils.OakSingleModelRunner import OakSingleModelRunner
from utils.draw import drawROI, displayFPS


def main():
    # Set few parameters
    nn_path = str(Path(__file__).parent) + "/../../models/coronamask.blob"
    labels = ["background", "no mask", "mask", "no mask"]

    # Init pipeline
    runner = OakSingleModelRunner() 

    # Configure middle camera and init output streams
    runner.setMiddleCamera(frame_width=300, frame_height=300)
    runner.middle_cam.setInterleaved(False)
    runner.middle_cam.setFps(20)

    # Configure stereo depth
    runner.setDepth()
    runner.stereo.setOutputDepth(True)
    runner.stereo.setConfidenceThreshold(255)

    # Configure neural network model and init input / output streams
    runner.setMobileNetDetectionModel(path=nn_path)
    runner.nn.setDepthLowerThreshold(250)
    runner.nn.setDepthUpperThreshold(5000)
    runner.labels = labels

    # Run the loop that call the process function
    runner.run(process=process, middle_cam_queue_size=4, nn_queue_size=4)


# Process function
def process(runner):
    frame = runner.middle_cam_output_queue.get().getCvFrame()
    detections = runner.nn_output_queue.get().detections
    
    for detection in detections:
        drawROI(frame, (detection.xmin,detection.ymin), (detection.xmax,detection.ymax), label=runner.labels[detection.label], confidence=detection.confidence, spatialCoordinates=detection.spatialCoordinates)
    displayFPS(frame, runner.getFPS())
    cv2.imshow("output", frame)


if __name__ == "__main__":
    main()