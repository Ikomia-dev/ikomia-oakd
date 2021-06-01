import cv2
import depthai as dai
import psutil
from matplotlib import pyplot as plt
from pathlib import Path
import sys
import numpy as np
import time



# return all needed oakD info
def get_sys_info(info):
    m = 1024 * 1024 # MiB
    drr = (info.ddrMemoryUsage.used / m)/(info.ddrMemoryUsage.total / m)
    cmx = (info.cmxMemoryUsage.used / m)/(info.cmxMemoryUsage.total / m)
    leonOS = info.leonCssCpuUsage.average * 100
    leonRT = info.leonMssCpuUsage.average * 100
    return drr, cmx, leonOS, leonRT



# Tiny yolo v3 label texts
labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]

syncNN = True



# Get argument first
nnPath = str((Path(__file__).parent / Path('../../_models/devices.blob')).resolve().absolute())
if len(sys.argv) > 1:
    nnPath = sys.argv[1]


# Program parameters
logger_freq = 15
fps_limit = 15
frame_width = 416
frame_height = 416

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera (RGB)
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(frame_width, frame_height)
camRgb.setInterleaved(False)
camRgb.setFps(fps_limit)

# Define system logger source
sys_logger = pipeline.createSystemLogger()
sys_logger.setRate(logger_freq)


# network specific settings
detectionNetwork = pipeline.createYoloDetectionNetwork()
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.setNumClasses(80)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
detectionNetwork.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
detectionNetwork.setIouThreshold(0.5)

detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

camRgb.preview.link(detectionNetwork.input)


# Create outputs
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
if syncNN:
    detectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

nnOut = pipeline.createXLinkOut()
nnOut.setStreamName("detections")
detectionNetwork.out.link(nnOut.input)

# Create outputs
linkOut = pipeline.createXLinkOut()
linkOut.setStreamName("sysinfo")
sys_logger.out.link(linkOut.input)


# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    q_sysinfo = device.getOutputQueue(name="sysinfo", maxSize=4, blocking=False)

    frame = None
    detections = []

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.imshow(name, frame)

    startTime = time.monotonic()
    counter = 0

    oakD_drr = []
    oakD_cmx = []
    oakD_leonOS = []
    oakD_leonRT = []
    host_ram = []
    host_cpu = []

    while True:
        if syncNN:
            inRgb = qRgb.get()
            inDet = qDet.get()
        else:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

        if inDet is not None:
            detections = inDet.detections
            counter += 1
            info = q_sysinfo.get()
            drr, cmx, leonOS, leonRT = get_sys_info(info)
            oakD_drr.append(drr)
            oakD_cmx.append(cmx)
            oakD_leonOS.append(leonOS)
            oakD_leonRT.append(leonRT)

            host_ram.append(psutil.virtual_memory().percent)
            host_cpu.append(psutil.cpu_percent())

        if frame is not None:
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
    
    
    time_range = [i/logger_freq for i in range(counter)]
    titles = ["Host RAM", "Host CPU", "OAK-D DRR", "OAK-D LeonOS", "OAK-D CMX", "OAK-D LeonRT"]
    infos = [host_ram, host_cpu, oakD_drr, oakD_leonOS, oakD_cmx, oakD_leonRT]

    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.title(titles[i])
        plt.plot(time_range, infos[i])
        plt.ylim(0, 100)
        plt.ylabel("usage (%)")
        plt.xlabel("time (s)")
    plt.show()