from pathlib import Path
import depthai as dai
import numpy as np
import cv2
import sys

# Importing from parent folder
sys.path.insert(0, str(Path(__file__).parent.parent.parent)) # move to parent path
from utils.compute import to_planar, get_landmark_3d, get_vector_intersection
from utils.visualize import LandmarksCubeVisualizer, LandmarksDepthVisualizer
from utils.draw import drawROI, displayFPS
from utils.OakRunner import OakRunner



frame_width, frame_height = 300, 300
visualizeLandmarksCube = True # define visualization mode (cube with centered face or cameras fields of views)



# Function called before entering inside the process loop, useful to set few arguments
def init(runner, device):
    runner.custom_arguments["required_confidence"] = 0.2

    colors = [(255,255,255), (255,255,255), (0,255,255), (255,0,255), (255,0,255)]
    pairs = [(0,2), (1,2), (3,4)]
    if(visualizeLandmarksCube):
        runner.custom_arguments["visualizer"] = LandmarksCubeVisualizer(window_width=frame_width, window_height=frame_height, cameras_positions=[runner.left_camera_location, runner.right_camera_location], size=1, colors=colors, pairs=pairs)
    else:
        runner.custom_arguments["visualizer"] = LandmarksDepthVisualizer(frame_width*2, frame_height, [runner.left_camera_location, runner.right_camera_location], colors=colors, pairs=pairs)
    runner.custom_arguments["visualizer"].start()


# Function called inside the process loop, useful to apply any treatment
def process(runner):
    spatial_vectors = dict()
    for side in ["left", "right"]:
        spatial_vectors[side] = []
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
            # print(landmarks)

            # Draw detections
            drawROI(frame, (xmin,ymin), (xmax,ymax), color=(0,200,230))
            for x,y in landmarks:
                cv2.circle(frame, (int(x*(xmax-xmin))+xmin,int(y*(ymax-ymin))+ymin), 2, (0,0,255))

            # Set spatial vectors
            spatial_landmarks = [get_landmark_3d((x,y)) for x,y in landmarks]
            camera_location = runner.left_camera_location if(side == "left") else runner.right_camera_location
            
            for i in range(5):
                spatial_vectors[side].append([spatial_landmarks[i][j] - camera_location[j] for j in range(3)])
            spatial_vectors[side] = np.array(spatial_vectors[side]) # convert list to numpy array

        displayFPS(frame, runner.getFPS())
        cv2.imshow(side, frame)

    # Determined depth to precisely locate landmarks in space
    landmark_spatial_locations = []
    if(len(spatial_vectors["left"])>4 and len(spatial_vectors["right"])>4):
        for i in range(5):
            landmark_spatial_locations.append(get_vector_intersection(spatial_vectors["left"][i], runner.left_camera_location, spatial_vectors["right"][i], runner.right_camera_location))

        runner.custom_arguments["visualizer"].setLandmarks(landmark_spatial_locations)



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

    runner.addNeuralNetworkModel(stream_name="nn_"+side+"_landmarks", path=str(Path(__file__).parent) + "/../../../_models/tiny_face_landmarks.blob", handle_mono_depth=False)

runner.run(process=process, init=init)