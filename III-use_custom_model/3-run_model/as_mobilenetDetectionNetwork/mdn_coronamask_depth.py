import depthai as dai
import time
import cv2
from pathlib import Path



# Draw ROI and class label of each detected thing
def frame_process(frame, detection):
    topleft = (int(detection.xmin*frame_width), int(detection.ymin*frame_height))
    bottomright = (int(detection.xmax*frame_width), int(detection.ymax*frame_height))
    color = (255,0,0)
    cv2.rectangle(frame, topleft, bottomright, color, 2) # ROI
    cv2.putText(frame, labels[detection.label] + f" {int(detection.confidence * 100)}%", (topleft[0] + 10, topleft[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color) # Label and confidence
    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (topleft[0] + 10, topleft[1] + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (topleft[0] + 10, topleft[1] + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (topleft[0] + 10, topleft[1] + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
    return frame



# Define program parameters
nn_path = str(Path(__file__).parent) + "/../models/coronamask.blob" # path to the neural network compiled model (.blob)
labels = ["background", "no mask", "mask", "no mask"]
pipeline = dai.Pipeline()
frame_width = 300
frame_height = 300
fps_limit = 20

# Prepare depth handling
depth = pipeline.createStereoDepth()

# Set depth source
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)


# Set rgb camera source
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(frame_width, frame_height)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(fps_limit)


# Configure neural network settings
nn = pipeline.createMobileNetSpatialDetectionNetwork()
nn.setConfidenceThreshold(0.5) # keep detections if confidence>50%
nn.setBlobPath(nn_path)
nn.setNumInferenceThreads(2)
cam_rgb.preview.link(nn.input) # link cam_rgb to nn input layer


# Set depth output stream
left.out.link(depth.left)
right.out.link(depth.right)
depth.depth.link(nn.inputDepth)

# Set rgb output stream
rgb_output_stream = pipeline.createXLinkOut()
rgb_output_stream.setStreamName("rgb")
nn.passthrough.link(rgb_output_stream.input)

# Set neural network output stream
nn_output_stream = pipeline.createXLinkOut()
nn_output_stream.setStreamName("nn")
nn.out.link(nn_output_stream.input)



with dai.Device(pipeline) as device:
    device.startPipeline()
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detection_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic() # To determined FPS
    counter = 0


    while True:
        rgb_current_output = rgb_queue.get()

        if rgb_current_output is not None:
            frame = rgb_current_output.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

            detections = detection_queue.get().detections
            for detection in detections:
                frame = frame_process(frame, detection) # process the frame
            counter += 1

        if frame is not None:
            cv2.imshow("output", frame)

        if cv2.waitKey(1) == ord('q'):
            break