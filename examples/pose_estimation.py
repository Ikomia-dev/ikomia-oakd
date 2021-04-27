from utils.pose import getKeypoints, getPersonwiseKeypoints, getValidPairs
from utils.OakSingleModelRunner import OakSingleModelRunner
from utils.draw import displayFPS
from pathlib import Path
import depthai as dai
import numpy as np
import cv2


# Define program parameters
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28],
          [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]]
colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
threshold = 0.3
nPoints = 18


# Process function
def process(runner):
    frame_width = runner.middle_cam.getPreviewWidth()
    frame_height = runner.middle_cam.getPreviewHeight()
    frame = runner.middle_cam_output_queue.get().getCvFrame()
    nn_current_output = runner.nn_output_queue.get()
    
    heatmaps = np.array(nn_current_output.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57)).astype('float32')
    pafs = np.array(nn_current_output.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57)).astype('float32')
    outputs = np.concatenate((heatmaps, pafs), axis=1)

    new_keypoints = []
    new_keypoints_list = np.zeros((0, 3))
    keypoint_id = 0

    # Process the model output to get proper outputs (keypoints and pairs)
    for row in range(nPoints):
        probMap = outputs[0, row, :, :]
        probMap = cv2.resize(probMap, (frame_width, frame_height))  # (456, 256)
        keypoints = getKeypoints(probMap, threshold)
        new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
        keypoints_with_id = []

        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoint_id += 1
        new_keypoints.append(keypoints_with_id)

    valid_pairs, invalid_pairs = getValidPairs(outputs, frame_width, frame_height, new_keypoints)
    newPersonwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, new_keypoints_list)
    detected_keypoints, keypoints_list, personwiseKeypoints = (new_keypoints, new_keypoints_list, newPersonwiseKeypoints)

    # Draw keypoints and pairs
    if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
        for i in range(nPoints):
            for j in range(len(detected_keypoints[i])):
                cv2.circle(frame, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
        for i in range(nPoints-1):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

    displayFPS(frame, runner.getFPS())
    cv2.imshow("output", frame)


runner = OakSingleModelRunner() 

# Configure middle camera and init output streams
runner.setMiddleCamera(frame_width=456, frame_height=256)
runner.middle_cam.setInterleaved(False)
runner.middle_cam.setFps(8)

# Configure neural network model and init input / output streams
runner.setNeuralNetworkModel(path=str(Path(__file__).parent) + "/../models/pose_estimation.blob")

# Run the loop that call the process function
runner.run(process=process, middle_cam_queue_size=1, nn_queue_size=1)