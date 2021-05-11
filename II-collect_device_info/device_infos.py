import cv2
import depthai as dai
import psutil
from matplotlib import pyplot as plt



# return all needed oakD info
def get_sys_info(info):
    m = 1024 * 1024 # MiB
    drr = (info.ddrMemoryUsage.used / m)/(info.ddrMemoryUsage.total / m)
    cmx = (info.cmxMemoryUsage.used / m)/(info.cmxMemoryUsage.total / m)
    leonOS = info.leonCssCpuUsage.average * 100
    leonRT = info.leonMssCpuUsage.average * 100
    return drr, cmx, leonOS, leonRT


# Most basic process, show the frame
def test_process(frame):
    cv2.imshow("output", frame)



# Program parameters
logger_freq = 15
fps_limit = 15
frame_width = 416
frame_height = 416

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera (RGB)
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(frame_width, frame_height)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRgb.setFps(fps_limit)

# Define system logger source
sys_logger = pipeline.create(dai.node.SystemLogger)
sys_logger.setRate(logger_freq)

# Create outputs
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)
linkOut = pipeline.create(dai.node.XLinkOut)
linkOut.setStreamName("sysinfo")
sys_logger.out.link(linkOut.input)


with dai.Device(pipeline) as device:
    device.startPipeline()
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_sysinfo = device.getOutputQueue(name="sysinfo", maxSize=4, blocking=False)

    oakD_drr = []
    oakD_cmx = []
    oakD_leonOS = []
    oakD_leonRT = []
    host_ram = []
    host_cpu = []
    counter = 0
    while True:
        inRgb = qRgb.get()
        test_process(inRgb.getCvFrame())

        info = q_sysinfo.get()
        drr, cmx, leonOS, leonRT = get_sys_info(info)
        oakD_drr.append(drr)
        oakD_cmx.append(cmx)
        oakD_leonOS.append(leonOS)
        oakD_leonRT.append(leonRT)

        host_ram.append(psutil.virtual_memory().percent)
        host_cpu.append(psutil.cpu_percent())

        counter += 1
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