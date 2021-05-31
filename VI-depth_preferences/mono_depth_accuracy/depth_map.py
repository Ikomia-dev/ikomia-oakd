from utils.OakRunner import OakRunner
from utils.draw import displayFPS
import depthai as dai
import cv2


def process(runner):
    frame = runner.output_queues["depth"].get().getFrame()
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    displayFPS(frame, runner.getFPS())
    cv2.imshow("disparity", frame)


runner = OakRunner()
runner.setMonoDepth(sensor_resolution=dai.MonoCameraProperties.SensorResolution.THE_400_P)
runner.run(process=process)