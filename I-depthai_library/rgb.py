import cv2
import depthai as dai

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera (RGB)
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(800, 540)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB) # Specify which board socket to use
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Create output
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input) # Link video stream to output

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    device.startPipeline() # Start video
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False) # Get camera output stream

    while True:
        inRgb = qRgb.get() # For each frame
        cv2.imshow("output", inRgb.getCvFrame()) # Show it

        if cv2.waitKey(1) == ord('q'):
            break