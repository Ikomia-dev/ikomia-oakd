import depthai as dai
import time
import cv2
from pathlib import Path



# Draw ROI and class label of each detected thing if confidence>60%
def frame_process(frame, detection):
    frame_width, frame_height = frame.shape[:2]
    topleft = (int(detection.xmin*frame_width), int(detection.ymin*frame_height))
    bottomright = (int(detection.xmax*frame_width), int(detection.ymax*frame_height))
    bottomleft = (int(detection.xmin*frame_width), int(detection.ymax*frame_height))
    cv2.rectangle(frame, topleft, bottomright, (255,0,0), 2) # ROI
    cv2.putText(frame, labels[detection.label], topleft, cv2.FONT_HERSHEY_TRIPLEX, 1.5, (200,0,160)) # Label
    cv2.putText(frame, f"{int(detection.confidence * 100)}%", bottomleft, cv2.FONT_HERSHEY_TRIPLEX, 1, (255,0,0)) # Confidence
    return frame



# Define program parameters
nn_path = str(Path(__file__).parent) + "/model.blob" # path to the neural network compiled model (.blob)
labels = ["background", "no mask", "mask", "no mask"]
pipeline = dai.Pipeline()

# Define sources  (middle camera)
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(20)

# Configure neural network settings
nn = pipeline.createMobileNetDetectionNetwork()
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(nn_path)
nn.setNumInferenceThreads(2)
cam_rgb.preview.link(nn.input) # link cam_rgb to nn input layer

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
        detections = detection_queue.get().detections

        if rgb_current_output is not None:
            frame = rgb_current_output.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

            for detection in detections:
                frame = frame_process(frame, detection) # process the frame
            counter += 1

        if frame is not None:
            cv2.imshow("output", frame)

        if cv2.waitKey(1) == ord('q'):
            break