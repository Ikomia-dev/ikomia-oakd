from pathlib import Path
import depthai as dai
import cv2


pipeline = dai.Pipeline()

# Configure left cam
left_cam = pipeline.createMonoCamera()
left_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
left_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

# Convert left cam output into proper sized image
manip = pipeline.createImageManip()
manip.initialConfig.setResize(300, 300)
left_cam.out.link(manip.inputImage)

# Set output stream
left_cam_manip_output_stream = pipeline.createXLinkOut()
left_cam_manip_output_stream.setStreamName("manip")
manip.out.link(left_cam_manip_output_stream.input)


with dai.Device(pipeline) as device:
    device.startPipeline()

    left_cam_manip_output_queue = device.getOutputQueue("manip", maxSize=4, blocking=False)

    while True:
        frame = left_cam_manip_output_queue.get().getCvFrame()
        cv2.imshow("output", frame)

        if cv2.waitKey(1) == ord('q'):
            break