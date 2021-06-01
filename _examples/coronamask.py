from utils.draw import drawROI, displayFPS
from utils.OakRunner import OakRunner
from pathlib import Path
import cv2


def process(runner):
    frame = runner.output_queues["middle_cam"].get().getCvFrame()
    detections = runner.output_queues["nn"].get().detections
    
    for det in detections:
        drawROI(frame, (det.xmin,det.ymin), (det.xmax,det.ymax),label=runner.labels[det.label], confidence=det.confidence)
    displayFPS(frame, runner.getFPS())
    cv2.imshow("output", frame)


runner = OakRunner() 

# Configure middle camera
runner.setMiddleCamera(frame_width=300, frame_height=300, stream_name="middle_cam")
middle_cam = runner.getMiddleCamera()
middle_cam.setInterleaved(False)
middle_cam.setFps(20)

# Configure neural network
runner.addMobileNetDetectionModel(stream_name="nn", path=str(Path(__file__).parent) + "/../models/coronamask.blob", handle_mono_depth=False)
middle_cam.preview.link(runner.neural_networks["nn"].input)
runner.labels = ["background", "no mask", "mask", "no mask"]

# Start the process loop
runner.run(process=process)