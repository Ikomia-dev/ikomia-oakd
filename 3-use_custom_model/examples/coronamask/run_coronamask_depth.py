import depthai as dai
import time
import cv2
from pathlib import Path



# Draw ROI and class label of each detected thing if confidence>50%
def frame_process(frame, tensor):
    color = (255,0,0)
    keeped_roi = []
    for i in range(100): # There is 100 detections, not all of them are relevant
        if (tensor[i*7 + 2] >0.5): # 3rd value of each detection is the confidence
            keeped_roi.append(tensor[i*7:i*7+7])

    spatial_calculator_config = dai.SpatialLocationCalculatorConfig()
    for  id, label, confidence, left, top, right, bottom  in  keeped_roi:
        topleft = (int(left*frame_width), int(top*frame_height))
        bottomright = (int(right*frame_width), int(bottom*frame_height))
        cv2.rectangle(frame, topleft, bottomright, color, 2) # ROI
        cv2.putText(frame, labels[int(label)] + f" {int(confidence * 100)}%", (topleft[0] + 10, topleft[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color) # Label and confidence
        
        # Add ROIs to spatial location calculator config
        spatial_config_data = dai.SpatialLocationCalculatorConfigData()
        spatial_config_data.roi = dai.Rect(dai.Point2f(topleft[0], topleft[1]), dai.Point2f(bottomright[0], bottomright[1]))
        spatial_calculator_config.addROI(spatial_config_data)

    # Put spatial location info inside of the ROI
    if(len(keeped_roi)>0):
        spatial_config_input_queue.send(spatial_calculator_config)
        spatial_data = spatial_calculator_queue.get().getSpatialLocations()
        for depth_data in spatial_data:
            cv2.putText(frame, f"X: {int(depth_data.spatialCoordinates.x)} mm", (topleft[0] + 10, topleft[1] + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Y: {int(depth_data.spatialCoordinates.y)} mm", (topleft[0] + 10, topleft[1] + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Z: {int(depth_data.spatialCoordinates.z)} mm", (topleft[0] + 10, topleft[1] + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
    return frame



# Define program parameters
nn_path = str(Path(__file__).parent) + "/model.blob" # path to the neural network compiled model (.blob)
labels = ["background", "no mask", "mask", "no mask"]
pipeline = dai.Pipeline()
frame_width = 300
frame_height = 300
fps_limit = 20


# Configure spatial location calculator
spatial_location_calculator = pipeline.createSpatialLocationCalculator()
spatial_location_calculator.setWaitForConfigInput(True)

# Prepare depth handling
depth = pipeline.createStereoDepth()
depth.depth.link(spatial_location_calculator.inputDepth)

# Set spatial location calculator input/output stream
spatial_data_output_stream = pipeline.createXLinkOut()
spatial_data_output_stream.setStreamName("spatialData")
spatial_location_calculator.out.link(spatial_data_output_stream.input)
spatial_config_input_stream = pipeline.createXLinkIn()
spatial_config_input_stream.setStreamName("spatialCalcConfig")
spatial_config_input_stream.out.link(spatial_location_calculator.inputConfig)


# Set rgb camera source
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(frame_width, frame_height)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(fps_limit)

# Set depth source
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)


# Configure neural network settings
nn = pipeline.createNeuralNetwork()
nn.setBlobPath(nn_path)
cam_rgb.preview.link(nn.input) # link cam_rgb to nn input layer


# Set rgb output stream
rgb_output_stream = pipeline.createXLinkOut()
rgb_output_stream.setStreamName("rgb")
nn.passthrough.link(rgb_output_stream.input)

# Set depth output stream
left.out.link(depth.left)
right.out.link(depth.right)

# Set neural network output stream
nn_output_stream = pipeline.createXLinkOut()
nn_output_stream.setStreamName("nn")
nn.out.link(nn_output_stream.input)



with dai.Device(pipeline) as device:
    device.startPipeline()
    spatial_config_input_queue = device.getInputQueue("spatialCalcConfig")
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    nn_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    spatial_calculator_queue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)

    frame = None
    startTime = time.monotonic() # To determined FPS
    counter = 0


    while True:
        rgb_current_output = rgb_queue.get()
        nn_current_output = nn_queue.get()

        if rgb_current_output is not None:
            frame = rgb_current_output.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

            # Process the data thanks to the NNData object
            if nn_current_output is not None:
                tensor = nn_current_output.getLayerFp16("DetectionOutput") # Get detection tensor (output layer "DetectionOutput" with this model)
                frame_process(frame, tensor)
                counter += 1

        if frame is not None:
            cv2.imshow("output", frame)

        if cv2.waitKey(1) == ord('q'):
            break