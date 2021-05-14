from pathlib import Path
import depthai as dai
import numpy as np
import cv2
import sys

# Importing from parent folder
sys.path.insert(0, str(Path(__file__).parent.parent.parent)) # move to parent path
from utils.draw import drawROI, displayFPS
from utils.OakRunner import OakRunner


def main():
    # Set few parameters
    nn_path = str(Path(__file__).parent) + "/../../../models/devices.blob"
    labels = [
        "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
        "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
        "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
        "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
        "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
        "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
        "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
        "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
        "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
        "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
        "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
        "teddy bear",     "hair drier", "toothbrush"
    ]
    anchor_masks = {"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])}
    anchors = np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
    coordinate_size = 4


    # Init pipeline
    runner = OakRunner()

    # Configure middle camera and init output streams
    runner.setMiddleCamera(frame_width=416, frame_height=416, stream_name="middle_cam", output_queue_size=4, block_output_queue=False)
    middle_cam = runner.getMiddleCamera()
    middle_cam.setInterleaved(False)
    middle_cam.setFps(20)

    # Configure stereo depth
    runner.setMonoDepth()
    stereo = runner.getStereo()
    stereo.setConfidenceThreshold(255)

    # Configure neural network model and init input / output streams
    runner.addYoloDetectionModel(stream_name="nn", path=nn_path, num_classes=len(labels), coordinate_size=coordinate_size, anchors=anchors, anchor_masks=anchor_masks, output_queue_size=4, handle_mono_depth=True)
    runner.labels = labels
    runner.neural_networks["nn"].setIouThreshold(0.5)
    runner.neural_networks["nn"].setDepthLowerThreshold(250)
    runner.neural_networks["nn"].setDepthUpperThreshold(5000)
    middle_cam.preview.link(runner.neural_networks["nn"].input)

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