from utils.OakSingleModelRunner import OakSingleModelRunner
from utils.draw import drawROI, displayFPS
from pathlib import Path
import cv2


def process(runner):
    frame = runner.middle_cam_output_queue.get().getCvFrame()
    detections = runner.nn_output_queue.get().detections
    
    for det in detections:
        drawROI(frame, (det.xmin,det.ymin), (det.xmax,det.ymax),label=runner.labels[det.label], confidence=det.confidence)
    displayFPS(frame, runner.getFPS())
    cv2.imshow("output", frame)


runner = OakSingleModelRunner() 

# Configure middle camera
runner.setMiddleCamera(frame_width=300, frame_height=300)
runner.middle_cam.setInterleaved(False)
runner.middle_cam.setFps(20)

# Configure neural network
runner.setMobileNetDetectionModel(path=str(Path(__file__).parent) + "/../models/coronamask.blob")
runner.labels = ["background", "no mask", "mask", "no mask"]

# Start the process loop
runner.run(process=process)