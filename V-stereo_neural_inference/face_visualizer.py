from utils.compute import to_planar, get_landmark_3d, get_vector_intersection
from utils.visualize import LandmarkCubeVisualizer, LandmarkDepthVisualizer
from utils.draw import displayFPS, drawROI
from pathlib import Path
import depthai as dai
import numpy as np
import time
import cv2



# Define program parameters
nn_face_detection_path = str(Path(__file__).parent) + "/../models/face_detection.blob"
nn_landmarks_path = str(Path(__file__).parent) + "/../models/landmarks.blob"
camera_locations = dict() # 3D location of cameras (for OAK-D)
camera_locations["left"] = (0.107, -0.038, 0.008)
camera_locations["right"] = (0.109, 0.039, 0.008)
visualizeLandmarksCube = True # define visualization mode (cube with centered face or cameras fields of views)

# Define neural network input sizes
face_input_width = 300 # also frame width
face_input_height = 300 # also frame height
landmarks_input_width = 48
landmarks_input_height = 48

# Init pipeline
pipeline = dai.Pipeline()


# Configure sided cameras and neural networks
for side in ["left", "right"]:
    # Init sided camera
    cam = pipeline.createMonoCamera()
    if(side == "left"):
        cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    # Transform camera output into proper face detection input
    face_manip = pipeline.createImageManip()
    face_manip.initialConfig.setResize(face_input_width, face_input_height)
    face_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p) # Switch to BGR (but still grayscaled)
    cam.out.link(face_manip.inputImage)

    # Set sided camera output stream
    face_manip_output_stream = pipeline.createXLinkOut()
    face_manip_output_stream.setStreamName(side + "_cam")
    face_manip.out.link(face_manip_output_stream.input)

    # Init the face detection neural network
    face_nn = pipeline.createNeuralNetwork()
    face_nn.setBlobPath(nn_face_detection_path)
    face_manip.out.link(face_nn.input)

    # Set face detection neural network output stream
    face_nn_output_stream = pipeline.createXLinkOut()
    face_nn_output_stream.setStreamName("nn_" + side + "_faces")
    face_nn.out.link(face_nn_output_stream.input)

    # Init the landmarks detection neural network
    landmarks_nn = pipeline.createNeuralNetwork()
    landmarks_nn.setBlobPath(nn_landmarks_path)

    # Set landmarks detection neural network input stream
    landmarks_nn_input_stream = pipeline.createXLinkIn()
    landmarks_nn_input_stream.setStreamName("nn_" + side + "_landmarks")
    landmarks_nn_input_stream.out.link(landmarks_nn.input)

    # Set landmarks detection neural network output stream
    landmarks_nn_output_stream = pipeline.createXLinkOut()
    landmarks_nn_output_stream.setStreamName("nn_" + side + "_landmarks")
    landmarks_nn.out.link(landmarks_nn_output_stream.input)


colors = [(255,255,255), (255,255,255), (0,255,255), (255,0,255), (255,0,255)]
pairs = [(0,2), (1,2), (3,4)]
if(visualizeLandmarksCube):
    visualizer = LandmarkCubeVisualizer(face_input_width, face_input_height, 1, [(0.107, -0.038, 0.008), (0.109, 0.039, 0.008)], colors, pairs)
else:
    visualizer = LandmarkDepthVisualizer(face_input_width*3, face_input_height, 2, [(0.107, -0.038, 0.008), (0.109, 0.039, 0.008)], colors, pairs)
visualizer.start()

with dai.Device(pipeline) as device:
    device.startPipeline()

    input_queues = dict()
    output_queues = dict()
    for side in ["left", "right"]:
        output_queues[side+"_cam"] = device.getOutputQueue(name=side+"_cam", maxSize=4, blocking=False)
        output_queues["nn_"+side+"_faces"] = device.getOutputQueue(name="nn_"+side+"_faces", maxSize=4, blocking=False)
        output_queues["nn_"+side+"_landmarks"] = device.getOutputQueue(name="nn_"+side+"_landmarks", maxSize=1, blocking=False)
        input_queues["nn_"+side+"_landmarks"] = device.getInputQueue(name="nn_"+side+"_landmarks", maxSize=1, blocking=False)

    spatial_vectors = dict()

    start_time = time.monotonic()
    frame_count = 0


    while True:
        for side in ["left", "right"]:
            spatial_vectors[side] = []
            frame = output_queues[side+"_cam"].get().getCvFrame()
            faces_data = output_queues["nn_"+side+"_faces"].get().getFirstLayerFp16()
            
            keeped_faces = []
            if(faces_data[2] > 0.2): # get most confident detection if > 20%
                keeped_faces.append([faces_data[3], faces_data[4], faces_data[5], faces_data[6]])

            # process faces
            for xmin, ymin, xmax, ymax in keeped_faces:
                # Get pixels instead of percentages
                xmin = int(xmin*face_input_width) if xmin>0 else 0
                xmax = int(xmax*face_input_width) if xmax<face_input_width else face_input_width
                ymin = int(ymin*face_input_height) if ymin>0 else 0
                ymax = int(ymax*face_input_height) if ymax<face_input_height else face_input_height

                # Compute the face to get landmarks
                land_data = dai.NNData()
                planar_cropped_face = to_planar(frame[ymin:ymax, xmin:xmax], (landmarks_input_width, landmarks_input_height))
                land_data.setLayer("0", planar_cropped_face)
                input_queues["nn_"+side+"_landmarks"].send(land_data)
                output = output_queues["nn_"+side+"_landmarks"].get().getFirstLayerFp16()
                landmarks = np.array(output).reshape(5,2)

                # Draw detections
                drawROI(frame, (xmin,ymin), (xmax,ymax), color=(0,200,230))
                for i in range(len(landmarks)):
                    cv2.circle(frame, (int(landmarks[i][0]*(xmax-xmin))+xmin,int(landmarks[i][1]*(ymax-ymin))+ymin), 2, (colors[i][2], colors[i][1], colors[i][0]))

                # Set spatial vectors
                spatial_landmarks = [get_landmark_3d((x,y)) for x,y in landmarks]
                for i in range(5):
                    spatial_vectors[side].append([spatial_landmarks[i][j] - camera_locations[side][j] for j in range(3)])
                spatial_vectors[side] = np.array(spatial_vectors[side]) # convert list to numpy array

            displayFPS(frame, (frame_count / (time.monotonic()-start_time)))
            cv2.imshow(side, frame)

        # Determined depth to precisely locate landmarks in space
        landmark_spatial_locations = []
        if(len(spatial_vectors["left"])>4 and len(spatial_vectors["right"])>4):
            for i in range(5):
                landmark_spatial_locations.append(get_vector_intersection(spatial_vectors["left"][i], camera_locations["left"], spatial_vectors["right"][i], camera_locations["right"]))

            visualizer.setLandmarks(landmark_spatial_locations)

        frame_count += 1
        if cv2.waitKey(1) == ord('q'):
            break