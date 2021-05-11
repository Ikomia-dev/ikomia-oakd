from pathlib import Path
import depthai as dai
import cv2
import sys

# Importing from parent folder
sys.path.insert(0, str(Path(__file__).parent.parent.parent)) # move to parent path
from utils.compute import updateSpatialCalculatorConfig
from utils.draw import drawROI, displayFPS
from utils.OakRunner import OakRunner


def main():
    # Set few parameters
    nn_path = str(Path(__file__).parent) + "/../../../models/coronamask.blob"
    labels = ["background", "no mask", "mask", "no mask"]

    # Init pipeline
    runner = OakRunner() 

    # Configure middle camera and init output streams
    runner.setMiddleCamera(frame_width=300, frame_height=300, stream_name="middle_cam", output_queue_size=4, block_output_queue=False)
    middle_cam = runner.getMiddleCamera()
    middle_cam.setInterleaved(False)
    middle_cam.setFps(20)

    # Configure stereo depth
    runner.setMonoDepth()
    stereo = runner.getStereo()
    stereo.setOutputDepth(True)
    stereo.setConfidenceThreshold(255)

    # Configure neural network model and init input / output streams
    runner.addNeuralNetworkModel(stream_name="nn", path=nn_path, output_queue_size=4, handle_mono_depth=True)
    middle_cam.preview.link(runner.neural_networks["nn"].input)
    runner.labels = labels
    slc = runner.getSpatialLocationCalculator()
    slc.setWaitForConfigInput(True)

    # Run the loop that call the process function
    runner.run(process=process)


# Process function
def process(runner):
    frame = runner.output_queues["middle_cam"].get().getCvFrame()
    tensor = runner.output_queues["nn"].get().getLayerFp16("DetectionOutput")
    
    keeped_roi = []
    for i in range(100): # There is 100 detections, not all of them are relevant
        if (tensor[i*7 + 2] >0.5): # 3rd value of each detection is the confidence
            keeped_roi.append(tensor[i*7:i*7+7])
    
    if(len(keeped_roi)>0):
        # Set spatial location config (input ROIs into the calculator)
        spatial_calculator_config = dai.SpatialLocationCalculatorConfig()
        for  id, label, confidence, left, top, right, bottom  in  keeped_roi:
            updateSpatialCalculatorConfig(spatial_calculator_config, dai.Point2f(left, top), dai.Point2f(right, bottom))

        # Draw infos
        runner.input_queues["slc"].send(spatial_calculator_config)
        spatial_data = runner.output_queues["slc"].get().getSpatialLocations()
        for i in range(len(keeped_roi)):
            drawROI(frame, (keeped_roi[i][3],keeped_roi[i][4]), (keeped_roi[i][5],keeped_roi[i][6]), label=runner.labels[int(keeped_roi[i][1])], confidence=keeped_roi[i][2], spatialCoordinates=spatial_data[i].spatialCoordinates)

    displayFPS(frame, runner.getFPS())
    cv2.imshow("output", frame)


if __name__ == "__main__":
    main()