from pathlib import Path
import depthai as dai
import cv2
import sys

# Importing from parent folder
sys.path.insert(0, str(Path(__file__).parent.parent)) # move to parent path
from utils.OakSingleModelRunner import OakSingleModelRunner
from utils.draw import drawROI, displayFPS


def main():
    # Set few parameters
    nn_path = str(Path(__file__).parent) + "/../../models/coronamask.blob"
    labels = ["background", "no mask", "mask", "no mask"]

    # Init pipeline
    runner = OakSingleModelRunner() 

    # Configure middle camera and init output streams
    runner.setMiddleCamera(frame_width=300, frame_height=300)
    runner.middle_cam.setInterleaved(False)
    runner.middle_cam.setFps(20)

    # Configure stereo depth
    runner.setDepth()

    # Configure neural network model and init input / output streams
    runner.setNeuralNetworkModel(path=nn_path)
    runner.labels = labels
    runner.spatial_location_calculator.setWaitForConfigInput(True)

    # Run the loop that call the process function
    runner.run(process=process, middle_cam_queue_size=4, nn_queue_size=4)


# Process function
def process(runner):
    frame = runner.middle_cam_output_queue.get().getCvFrame()
    tensor = runner.nn_output_queue.get().getLayerFp16("DetectionOutput")
    
    keeped_roi = []
    for i in range(100): # There is 100 detections, not all of them are relevant
        if (tensor[i*7 + 2] >0.5): # 3rd value of each detection is the confidence
            keeped_roi.append(tensor[i*7:i*7+7])
    
    if(len(keeped_roi)>0):
        # Set spatial location config (input ROIs into the calculator)
        spatial_calculator_config = dai.SpatialLocationCalculatorConfig()
        for  id, label, confidence, left, top, right, bottom  in  keeped_roi:
            spatial_config_data = dai.SpatialLocationCalculatorConfigData()
            spatial_config_data.roi = dai.Rect(dai.Point2f(left, top), dai.Point2f(right, bottom))
            spatial_calculator_config.addROI(spatial_config_data)

        # Draw infos
        runner.spatial_calculator_input_queue.send(spatial_calculator_config)
        spatial_data = runner.spatial_calculator_output_queue.get().getSpatialLocations()
        for i in range(len(keeped_roi)):
            drawROI(frame, (keeped_roi[i][3],keeped_roi[i][4]), (keeped_roi[i][5],keeped_roi[i][6]), label=runner.labels[int(keeped_roi[i][1])], confidence=keeped_roi[i][2], spatialCoordinates=spatial_data[i].spatialCoordinates)

    displayFPS(frame, runner.getFPS())
    cv2.imshow("output", frame)


if __name__ == "__main__":
    main()