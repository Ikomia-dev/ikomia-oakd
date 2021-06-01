from utils.compute import get_landmark_3d, get_vector_intersection
from utils.visualize import HumanPoseVisualizer
from utils.OakRunner import OakRunner
from utils.pose import getKeypoints
from utils.draw import displayFPS
from pathlib import Path
import depthai as dai
import numpy as np
import cv2



fps_limit = 3
frame_width, frame_height = 456, 256
pairs = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]
colors = [[255, 100, 0], [255, 100, 0], [255, 255, 0], [255, 100, 0], [255, 255, 0], [255, 100, 0], [0, 255, 0],
          [100, 200, 255], [255, 0, 255], [0, 255, 0], [100, 200, 255], [255, 0, 255], [255, 0, 0], [0, 0, 255],
          [0, 200, 200], [0, 0, 255], [0, 200, 200], [0, 0, 0]]
threshold = 0.3
nb_points = 18



def init(runner):
    runner.custom_arguments["visualizer"] = HumanPoseVisualizer(600, 600, [runner.left_camera_location, runner.right_camera_location], colors=colors, pairs=pairs)
    runner.custom_arguments["visualizer"].start()


def process(runner):
    spatial_vectors = dict()
    for side in ["left", "right"]:
        frame = runner.output_queues[side+"_cam"].get().getCvFrame()
        nn_current_output = runner.output_queues["nn_"+side].get()
        
        heatmaps = np.array(nn_current_output.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57)).astype('float32')
        pafs = np.array(nn_current_output.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57)).astype('float32')
        outputs = np.concatenate((heatmaps, pafs), axis=1)

        spatial_vectors[side] = []
        landmarks = []
        for i in range(nb_points):
            probMap = outputs[0, i, :, :]
            probMap = cv2.resize(probMap, (frame_width, frame_height))
            keypoints = getKeypoints(probMap, threshold)
            if(len(keypoints) > 0 and len(keypoints[0]) > 1):
                spatial_vectors[side].append(np.array(get_landmark_3d((keypoints[0][0]/frame_width, keypoints[0][1]/frame_height))))
                landmarks.append([keypoints[0][0], keypoints[0][1]])
                cv2.circle(frame, (keypoints[0][0], keypoints[0][1]), 5, (colors[i][2], colors[i][1], colors[i][0]), -1, cv2.LINE_AA) # draw keypoint
            else:
                spatial_vectors[side].append(keypoints) # insert empty array if the keypoint is not detected with enough confidence
                landmarks.append(keypoints)

        for pair in pairs:
            if(np.alltrue([len(landmarks[i])==2 for i in pair])):
                color = [0, 0, 0]
                for i in range(3):
                    color[i] += colors[pair[0]][i]/2
                    color[i] += colors[pair[1]][i]/2
                cv2.line(frame, (landmarks[pair[0]][0], landmarks[pair[0]][1]), (landmarks[pair[1]][0], landmarks[pair[1]][1]), (color[2], color[1], color[0]), 3, cv2.LINE_AA)

        displayFPS(frame, runner.getFPS())
        cv2.imshow(side, frame)

    # Determined depth to precisely locate landmarks in space
    landmark_spatial_locations = []
    for i in range(nb_points):
        landmark_spatial_locations.append(np.array(get_vector_intersection(spatial_vectors["left"][i], runner.left_camera_location, spatial_vectors["right"][i], runner.right_camera_location)))
    runner.custom_arguments["visualizer"].setLandmarks(landmark_spatial_locations)



runner = OakRunner() 

for side in ["left", "right"]:
    if(side == "left"):
        runner.setLeftCamera(frame_width, frame_height)
        runner.getLeftCamera().setFps(fps_limit)
        face_manip = runner.getLeftCameraManip()
    else:
        runner.setRightCamera(frame_width, frame_height)
        runner.getRightCamera().setFps(fps_limit)
        face_manip = runner.getRightCameraManip()

    face_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p) # Switch to BGR (but still grayscaled)
    runner.addNeuralNetworkModel(stream_name="nn_"+side, path=str(Path(__file__).parent) + "/../models/pose_estimation.blob", handle_mono_depth=False)
    face_manip.out.link(runner.neural_networks["nn_"+side].input) # link transformed video stream to neural network entry

runner.run(process=process, init=init)