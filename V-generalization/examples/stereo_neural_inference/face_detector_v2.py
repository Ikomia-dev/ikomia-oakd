from pathlib import Path
import depthai as dai
import cv2
import sys

# Importing from parent folder
sys.path.insert(0, str(Path(__file__).parent.parent.parent)) # move to parent path
from utils.draw import drawROI, displayFPS
from utils.OakRunner import OakRunner



frame_width, frame_height = 300, 300



# Function called before entering inside the process loop, useful to set few arguments
def init(runner, device):
    runner.custom_arguments["required_confidence"] = 0.2


# Function called inside the process loop, useful to apply any treatment
def process(runner):
     for side in ["left", "right"]:
        frame = runner.output_queues[side+"_cam"].get().getCvFrame()
        faces_data = runner.output_queues["nn_"+side+"_faces"].get().getFirstLayerFp16()

        if(faces_data[2] > runner.custom_arguments["required_confidence"]):
            drawROI(frame, (faces_data[3],faces_data[4]), (faces_data[5],faces_data[6]), color=(0,200,230))
        displayFPS(frame, runner.getFPS())
        cv2.imshow(side, frame)



runner = OakRunner()

for side in ["left", "right"]:
    if(side == "left"):
        runner.setLeftCamera(frame_width, frame_height)
        face_manip = runner.getLeftCameraManip()
    else:
        runner.setRightCamera(frame_width, frame_height)
        face_manip = runner.getRightCameraManip()

    face_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p) # Switch to BGR (but still grayscaled)
    runner.addNeuralNetworkModel(stream_name="nn_"+side+"_faces", path=str(Path(__file__).parent) + "/../../../_models/face_detection.blob", handle_mono_depth=False)
    face_manip.out.link(runner.neural_networks["nn_"+side+"_faces"].input) # link transformed video stream to neural network entry

runner.run(process=process, init=init)