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

    for  id, label, confidence, left, top, right, bottom  in  keeped_roi:
        topleft = (int(left*frame_width), int(top*frame_height))
        bottomright = (int(right*frame_width), int(bottom*frame_height))
        cv2.rectangle(frame, topleft, bottomright, color, 2) # ROI
        cv2.putText(frame, labels[int(label)] + f" {int(confidence * 100)}%", (topleft[0] + 10, topleft[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color) # Label and confidence
    return frame



# Define program parameters
nn_path = str(Path(__file__).parent) + "/../models/coronamask.blob" # path to the neural network compiled model (.blob)
labels = ["background", "no mask", "mask", "no mask"]
pipeline = dai.Pipeline()
frame_width = 300
frame_height = 300
fps_limit = 20


# Set rgb camera source
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(frame_width, frame_height)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(fps_limit)


# Configure neural network settings
nn = pipeline.createNeuralNetwork()
nn.setBlobPath(nn_path)
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
    nn_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

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
                frame = frame_process(frame, tensor) # Process the frame with the tensor (here: draw ROI / text label)
                counter += 1

        if frame is not None:
            cv2.imshow("output", frame)

        if cv2.waitKey(1) == ord('q'):
            break