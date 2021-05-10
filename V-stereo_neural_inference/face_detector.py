from utils.draw import displayFPS, drawROI
from pathlib import Path
import depthai as dai
import time
import cv2



# Define program parameters
nn_face_detection_path = str(Path(__file__).parent) + "/../models/face_detection.blob"

# Define neural network input sizes
face_input_width = 300 # also frame width
face_input_height = 300 # also frame height

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



with dai.Device(pipeline) as device:
    device.startPipeline()

    output_queues = dict()
    for side in ["left", "right"]:
        output_queues[side+"_cam"] = device.getOutputQueue(name=side+"_cam", maxSize=4, blocking=False)
        output_queues["nn_"+side+"_faces"] = device.getOutputQueue(name="nn_"+side+"_faces", maxSize=4, blocking=False)

    start_time = time.monotonic()
    frame_count = 0


    while True:
        for side in ["left", "right"]:
            frame = output_queues[side+"_cam"].get().getCvFrame()
            faces_data = output_queues["nn_"+side+"_faces"].get().getFirstLayerFp16()

            keeped_faces = []
            if(faces_data[2] > 0.2): # get most confident detection if > 20%
                keeped_faces.append([faces_data[3], faces_data[4], faces_data[5], faces_data[6]])

            for xmin, ymin, xmax, ymax in keeped_faces:
                drawROI(frame, (xmin,ymin), (xmax,ymax))

            displayFPS(frame, (frame_count / (time.monotonic() - start_time)))
            cv2.imshow(side, frame)

        frame_count += 1
        if cv2.waitKey(1) == ord('q'):
            break