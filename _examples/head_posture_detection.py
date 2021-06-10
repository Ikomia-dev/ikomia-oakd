from utils.compute import to_planar, frame_norm
from utils.OakRunner import OakRunner
from utils.pose import getHeadPose
from utils.draw import displayFPS
from pathlib import Path
import depthai as dai
import numpy as np
import cv2



frame_width = 300
frame_height = 300
fps_limit = 30



def process(runner):
    frame = runner.output_queues["middle_cam"].get().getCvFrame()
    faces_roi = np.array(runner.output_queues["nn_face"].get().getFirstLayerFp16())
    faces_roi = faces_roi.reshape((faces_roi.size // 7, 7)) # reorder data
    faces_roi = faces_roi[faces_roi[:, 2] > 0.7][:, 3:7] # keep if enought confidence (70%)

    for face_roi in faces_roi:
        xmin, ymin, xmax, ymax = frame_norm(frame, face_roi)
        face = frame[ymin:ymax, xmin:xmax]
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,255,0), 2)

        # Send cropped and resized face into landmarks detection network
        landmarks_input_data = dai.NNData()
        landmarks_input_data.setLayer("data", to_planar(face, (160,160)))
        runner.input_queues["nn_landmarks"].send(landmarks_input_data)
        
        landmarks = runner.output_queues["nn_landmarks"].get().getFirstLayerFp16()
        hand_points = [None]*14 # contain specific landmarks that will be use to determined head posture
        for i in range(int(len(landmarks)/2)):
            x, y = int(landmarks[i*2]*(xmax-xmin))+xmin, int(landmarks[i*2+1]*(ymax-ymin))+ymin
            cv2.circle(frame, (x, y), 1, (0,0,0), -1, cv2.LINE_AA) 
            if(i == 8): # Chin corner
                hand_points[13] = [landmarks[i*2]+xmin, landmarks[i*2+1]+ymin]
            elif(i == 17): # Left eyebrow upper left corner
                hand_points[0] = [landmarks[i*2]+xmin, landmarks[i*2+1]+ymin]
            elif(i == 21): # Left eyebrow right corner
                hand_points[1] = [landmarks[i*2]+xmin, landmarks[i*2+1]+ymin]
            elif(i == 22): # Right eyebrow upper left corner
                hand_points[2] = [landmarks[i*2]+xmin, landmarks[i*2+1]+ymin]
            elif(i == 26): # Right eyebrow upper right corner
                hand_points[3] = [landmarks[i*2]+xmin, landmarks[i*2+1]+ymin]
            elif(i == 31): # Upper left corner of the nose
                hand_points[8] = [landmarks[i*2]+xmin, landmarks[i*2+1]+ymin]
            elif(i == 35): # Upper right corner of the nose
                hand_points[9] = [landmarks[i*2]+xmin, landmarks[i*2+1]+ymin]
            elif(i == 36): # Left eye upper left corner
                hand_points[4] = [landmarks[i*2]+xmin, landmarks[i*2+1]+ymin]
            elif(i == 39): # Left eye upper right corner
                hand_points[5] = [landmarks[i*2]+xmin, landmarks[i*2+1]+ymin]
            elif(i == 42): # Right eye upper left corner
                hand_points[6] = [landmarks[i*2]+xmin, landmarks[i*2+1]+ymin]
            elif(i == 45): # Upper right corner of the right eye
                hand_points[7] = [landmarks[i*2]+xmin, landmarks[i*2+1]+ymin]
            elif(i == 48): # Upper left corner
                hand_points[10] = [landmarks[i*2]+xmin, landmarks[i*2+1]+ymin]
            elif(i == 54): # Upper right corner of the mouth
                hand_points[11] = [landmarks[i*2]+xmin, landmarks[i*2+1]+ymin]
            elif(i == 57): # Lower central corner of the mouth
                hand_points[12] = [landmarks[i*2]+xmin, landmarks[i*2+1]+ymin]

        try:
            reprojectdst, _, pitch, yaw, roll = getHeadPose(np.array(hand_points))
            cv2.putText(frame,"pitch:{:.2f}, yaw:{:.2f}, roll:{:.2f}".format(pitch,yaw,roll),(xmin-30,ymin-30),cv2.FONT_HERSHEY_COMPLEX,0.45,(255,0,0))
            line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
            for start, end in line_pairs:
                cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))
        except:
            pass

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