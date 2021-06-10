from utils.OakRunner import OakRunner
import depthai as dai
import numpy as np



def getProjectionMatrix(intrinsic_matrix, extrinsic_matrix):
    return intrinsic_matrix.dot(extrinsic_matrix)


def init(runner, device):
    calibration = device.readCalibration() # Object that contain camera intrisics and extrinsics

    left_intrinsics = np.array(calibration.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, 1280, 720))
    print("-----------------------------------\nIntrinsics (left camera)\n", left_intrinsics)

    right_intrinsics = np.array(calibration.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 1280, 720))
    print("\n-----------------------------------\nIntrinsics (right camera)\n", right_intrinsics)

    extrinsics = np.array(calibration.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT))[:3,:]
    print("\n-----------------------------------\nExtrinsics (both cameras)\n", extrinsics)

    P_left = getProjectionMatrix(left_intrinsics, extrinsics)
    print("\n-----------------------------------\nProjection (left camera)\n", P_left)

    P_right = getProjectionMatrix(right_intrinsics, extrinsics)
    print("\n-----------------------------------\nProjection (right camera)\n", P_right)


def process(runner):
    exit()



runner = OakRunner()
pipeline = runner.getPipeline()

runner.setRightCamera(1280, 720)
runner.setLeftCamera(1280, 720)

runner.run(process=process, init=init)