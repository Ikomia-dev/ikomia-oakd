from utils.compute import to_planar
from utils.OakRunner import OakRunner
from utils.draw import displayFPS
from pathlib import Path
import depthai as dai
import numpy as np
import cv2



frame_width = 300
frame_height = 300
fps_limit = 30



# Translate percentage coordinates into pixel coordinates
def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


def process(runner):
    frame = runner.output_queues["middle_cam"].get().getCvFrame()
    faces_roi = np.array(runner.output_queues["nn_face"].get().getFirstLayerFp16())
    faces_roi = faces_roi.reshape((faces_roi.size // 7, 7)) # reorder data
    faces_roi = faces_roi[faces_roi[:, 2] > 0.7][:, 3:7] # keep if enought confidence (70%)

    for face_roi in faces_roi:
        xmin, ymin, xmax, ymax = frame_norm(frame, face_roi)
        face = frame[ymin:ymax, xmin:xmax]
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,255,0), 2)

        landmarks_input_data = dai.NNData()
        landmarks_input_data.setLayer("data", to_planar(face, (160,160)))
        runner.input_queues["nn_landmarks"].send(landmarks_input_data)
        
        disordened_landmarks = runner.output_queues["nn_landmarks"].get().getFirstLayerFp16()
        landmarks = []
        for i in range(int(len(disordened_landmarks)/2)):
            x, y = int(disordened_landmarks[i*2]*(xmax-xmin)), int(disordened_landmarks[i*2+1]*(ymax-ymin))
            cv2.circle(frame, (x+xmin, y+ymin), 1, (0,0,0), -1, cv2.LINE_AA) 
            landmarks.append((x, y))

    displayFPS(frame, runner.getFPS())
    cv2.imshow("output", frame)



runner = OakRunner()

runner.setMiddleCamera(frame_width=frame_width, frame_height=frame_height)
cam = runner.getMiddleCamera()
cam.setInterleaved(False)
cam.setFps(fps_limit)

runner.addNeuralNetworkModel(stream_name="nn_face", path=str(Path(__file__).parent) + "/../_models/face_detection_v2.blob")
cam.preview.link(runner.neural_networks["nn_face"].input)
runner.addNeuralNetworkModel(stream_name="nn_landmarks", path=str(Path(__file__).parent) + "/../_models/face_landmarks.blob")

runner.run(process=process)