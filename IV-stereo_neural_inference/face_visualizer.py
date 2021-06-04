from visualize import LandmarksCubeVisualizer, LandmarksDepthVisualizer
from pathlib import Path
import depthai as dai
import numpy as np
import math
import time
import cv2



# Define program parameters
nn_face_detection_path = str(Path(__file__).parent) + "/../_models/face_detection.blob"
nn_landmarks_path = str(Path(__file__).parent) + "/../_models/landmarks.blob"
camera_locations = dict() # 3D location of cameras (for OAK-D)
camera_locations["left"] = (0.107, -0.038, 0.008)
camera_locations["right"] = (0.109, 0.039, 0.008)
visualizeLandmarksCube = True # define visualization mode (cube with centered face or cameras fields of views)

# Define neural network input sizes
face_input_width = 300 # also frame width
face_input_height = 300 # also frame height
landmarks_input_width = 48
landmarks_input_height = 48



# Convert multidimensional array into unidimensional array
def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


# Get vector from camera to landmark direction
def get_landmark_3d(landmark, focal_length=842, size=640):
    landmark_norm = 0.5 - np.array(landmark)
    landmark_image_coord = landmark_norm * size

    landmark_spherical_coord = [math.atan2(landmark_image_coord[0], focal_length),
                                -math.atan2(landmark_image_coord[1], focal_length) + math.pi / 2]

    landmarks_3D = [
        math.sin(landmark_spherical_coord[1]) * math.cos(landmark_spherical_coord[0]),
        math.sin(landmark_spherical_coord[1]) * math.sin(landmark_spherical_coord[0]),
        math.cos(landmark_spherical_coord[1])
    ]

    return landmarks_3D


# Get intersection point between two vectors
def get_vector_intersection(left_vector, left_camera_position, right_vector, right_camera_position):
    n = np.cross(left_vector, right_vector)
    n1 = np.cross(left_vector, n)
    n2 = np.cross(right_vector, n)

    top = np.dot(np.subtract(right_camera_position, left_camera_position), n2)
    bottom = np.dot(left_vector, n2)
    divided = top / bottom
    mult = divided * left_vector
    c1 = left_camera_position + mult

    top = np.dot(np.subtract(left_camera_position, right_camera_position), n1)
    bottom = np.dot(right_vector, n1)
    divided = top / bottom
    mult = divided * right_vector
    c2 = right_camera_position + mult

    center = (c1 + c2) / 2
    return center



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
    visualizer = LandmarksCubeVisualizer(face_input_width, face_input_height, [camera_locations["left"], camera_locations["right"]], colors=colors, pairs=pairs)
else:
    visualizer = LandmarksDepthVisualizer(face_input_width*2, face_input_height, [camera_locations["left"], camera_locations["right"]], colors=colors, pairs=pairs)
visualizer.start()

with dai.Device(pipeline) as device:
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
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,200,230), 2)
                for i in range(len(landmarks)):
                    cv2.circle(frame, (int(landmarks[i][0]*(xmax-xmin))+xmin,int(landmarks[i][1]*(ymax-ymin))+ymin), 2, (colors[i][2], colors[i][1], colors[i][0]))

                # Set spatial vectors
                spatial_landmarks = [get_landmark_3d((x,y)) for x,y in landmarks]
                for i in range(5):
                    spatial_vectors[side].append([spatial_landmarks[i][j] - camera_locations[side][j] for j in range(3)])
                spatial_vectors[side] = np.array(spatial_vectors[side]) # convert list to numpy array

            cv2.putText(frame, f"fps: {int((frame_count / (time.monotonic() - start_time)))}", (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))
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