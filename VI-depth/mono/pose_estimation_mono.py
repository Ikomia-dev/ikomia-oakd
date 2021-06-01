from utils.visualize import HumanPoseVisualizer
from utils.OakRunner import OakRunner
from utils.pose import getKeypoints
from utils.draw import displayFPS
from pathlib import Path
import depthai as dai
import numpy as np
import cv2



fps_limit = 6
frame_width, frame_height = 456, 256
pairs = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]
colors = [[255, 100, 0], [255, 100, 0], [255, 255, 0], [255, 100, 0], [255, 255, 0], [255, 100, 0], [0, 255, 0],
          [100, 200, 255], [255, 0, 255], [0, 255, 0], [100, 200, 255], [255, 0, 255], [255, 0, 0], [0, 0, 255],
          [0, 200, 200], [0, 0, 255], [0, 200, 200], [0, 0, 0]]
threshold = 0.3
nb_points = 18
min_depth = 250
max_depth = 2000



def init(runner, device):
    runner.custom_arguments["visualizer"] = HumanPoseVisualizer(300, 300, [runner.left_camera_location, runner.right_camera_location], size=10, colors=colors, pairs=pairs)
    runner.custom_arguments["visualizer"].start()


def process(runner):
    frame = runner.output_queues["middle_cam"].get().getCvFrame()
    nn_current_output = runner.output_queues["nn"].get()
    
    heatmaps = np.array(nn_current_output.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57)).astype('float32')
    pafs = np.array(nn_current_output.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57)).astype('float32')
    outputs = np.concatenate((heatmaps, pafs), axis=1)
    slc_conf = dai.SpatialLocationCalculatorConfig()

    landmarks = []
    spatial_landmarks = []
    for i in range(nb_points):
        probMap = outputs[0, i, :, :]
        probMap = cv2.resize(probMap, (frame_width, frame_height))
        keypoints = getKeypoints(probMap, threshold)
        if(len(keypoints) > 0 and len(keypoints[0]) > 1):
            landmarks.append([keypoints[0][0], keypoints[0][1]])
            cv2.circle(frame, (keypoints[0][0], keypoints[0][1]), 5, (colors[i][2], colors[i][1], colors[i][0]), -1, cv2.LINE_AA)

            slc_conf_data = dai.SpatialLocationCalculatorConfigData()
            slc_conf_data.depthThresholds.lowerThreshold = min_depth
            slc_conf_data.depthThresholds.upperThreshold = max_depth
            slc_conf_data.roi = dai.Rect(dai.Point2f(keypoints[0][0]-1, keypoints[0][1]-1), dai.Point2f(keypoints[0][0]+1, keypoints[0][1]+1))
            slc_conf.addROI(slc_conf_data)

            runner.input_queues["slc"].send(slc_conf)
            spatial_data = runner.output_queues["slc"].get().getSpatialLocations()
            spatial_landmarks.append([spatial_data[0].spatialCoordinates.x, spatial_data[0].spatialCoordinates.y, spatial_data[0].spatialCoordinates.z])
        else:
            landmarks.append(keypoints)
            spatial_landmarks.append(keypoints)

    for pair in pairs:
        if(np.alltrue([len(landmarks[i])==2 for i in pair])):
            color = [0, 0, 0]
            for i in range(3):
                color[i] += colors[pair[0]][i]/2
                color[i] += colors[pair[1]][i]/2
            cv2.line(frame, (landmarks[pair[0]][0], landmarks[pair[0]][1]), (landmarks[pair[1]][0], landmarks[pair[1]][1]), (color[2], color[1], color[0]), 3, cv2.LINE_AA)

    displayFPS(frame, runner.getFPS())
    cv2.imshow("output", frame)

    runner.custom_arguments["visualizer"].setLandmarks(spatial_landmarks)



runner = OakRunner() 

runner.setMiddleCamera(frame_width, frame_height)
cam = runner.getMiddleCamera()
cam.setFps(fps_limit)
cam.setInterleaved(False)

runner.setMonoDepth()

runner.addNeuralNetworkModel(stream_name="nn", path=str(Path(__file__).parent)+"/../../_models/pose_estimation.blob", handle_mono_depth=True)
cam.preview.link(runner.neural_networks["nn"].input)

slc = runner.getSpatialLocationCalculator()
slc.setWaitForConfigInput(True)

runner.run(process=process, init=init)