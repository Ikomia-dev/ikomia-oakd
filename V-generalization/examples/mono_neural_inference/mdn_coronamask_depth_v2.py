from pathlib import Path
import depthai as dai
import cv2
import sys

# Importing from parent folder
sys.path.insert(0, str(Path(__file__).parent.parent.parent)) # move to parent path
from utils.draw import drawROI, displayFPS
from utils.OakRunner import OakRunner


def main():
    # Set few parameters
    nn_path = str(Path(__file__).parent) + "/../../../_models/coronamask.blob"
    labels = ["background", "no mask", "mask", "no mask"]

    # Init pipeline
    runner = OakRunner() 

    # Configure middle camera and init output streams
    runner.setMiddleCamera(frame_width=300, frame_height=300, stream_name="middle_cam", output_queue_size=4, block_output_queue=False)
    middle_cam = runner.getMiddleCamera()
    middle_cam.setInterleaved(False)
    middle_cam.setFps(20)

    # Configure stereo depth
    runner.setMonoDepth()
    stereo = runner.getStereo()
    stereo.setConfidenceThreshold(255)

    # Configure neural network model and init input / output streams
    runner.addMobileNetDetectionModel(stream_name="nn", path=nn_path, output_queue_size=4, handle_mono_depth=True)
    runner.neural_networks["nn"].setDepthLowerThreshold(250)
    runner.neural_networks["nn"].setDepthUpperThreshold(5000)
    middle_cam.preview.link(runner.neural_networks["nn"].input)
    runner.labels = labels

    # Run the loop that call the process function
    runner.run(process=process)


# Process function
def process(runner):
    frame = runner.output_queues["middle_cam"].get().getCvFrame()
    detections = runner.output_queues["nn"].get().detections
    
    for detection in detections:
        drawROI(frame, (detection.xmin,detection.ymin), (detection.xmax,detection.ymax), label=runner.labels[detection.label], confidence=detection.confidence, spatialCoordinates=detection.spatialCoordinates)
    displayFPS(frame, runner.getFPS())
    cv2.imshow("output", frame)


if __name__ == "__main__":
    main()