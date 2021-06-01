from pathlib import Path
import depthai as dai
import numpy as np
import cv2
import sys

# Importing from parent folder
sys.path.insert(0, str(Path(__file__).parent.parent.parent)) # move to parent path
from utils.draw import drawROI, displayFPS
from utils.OakRunner import OakRunner
from utils.compute import to_planar



frame_width, frame_height = 300, 300



# Function called before entering inside the process loop, useful to set few arguments
def init(runner):
    runner.custom_arguments["required_confidence"] = 0.2


# Function called inside the process loop, useful to apply any treatment
def process(runner):
    for side in ["left", "right"]:
        frame = runner.output_queues[side+"_cam"].get().getCvFrame()
        faces_data = runner.output_queues["nn_"+side+"_faces"].get().getFirstLayerFp16()

        if(faces_data[2] > runner.custom_arguments["required_confidence"]):
            # Get pixels instead of percentages
            xmin = int(faces_data[3]*frame_width) if faces_data[3]>0 else 0
            ymin = int(faces_data[4]*frame_height) if faces_data[4]>0 else 0
            xmax = int(faces_data[5]*frame_width) if faces_data[5]<1 else frame_width
            ymax = int(faces_data[6]*frame_height) if faces_data[6]<1 else frame_height

            # Compute the face to get landmarks
            land_data = dai.NNData()
            planar_cropped_face = to_planar(frame[ymin:ymax, xmin:xmax], (48, 48))
            land_data.setLayer("0", planar_cropped_face)
            runner.input_queues["nn_"+side+"_landmarks"].send(land_data)
            output = runner.output_queues["nn_"+side+"_landmarks"].get().getFirstLayerFp16()
            landmarks = np.array(output).reshape(5,2)

            # Draw detections
            drawROI(frame, (xmin,ymin), (xmax,ymax), color=(0,200,230))
            for x,y in landmarks:
                cv2.circle(frame, (int(x*(xmax-xmin))+xmin,int(y*(ymax-ymin))+ymin), 2, (0,0,255))

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

    runner.addNeuralNetworkModel(stream_name="nn_"+side+"_landmarks", path=str(Path(__file__).parent) + "/../../../_models/landmarks.blob", handle_mono_depth=False)

runner.run(process=process, init=init)