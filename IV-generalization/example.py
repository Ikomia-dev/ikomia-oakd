from OakSingleModelRunner import OakSingleModelRunner
from pathlib import Path
import depthai as dai
import cv2


def main():
    # Set few parameters
    nn_path = str(Path(__file__).parent) + "/../models/coronamask.blob"
    labels = ["background", "no mask", "mask", "no mask"]

    # Init pipeline
    oak_detection = OakSingleModelRunner() 

    # Configure middle camera and init output streams
    oak_detection.setMiddleCamera(frame_width=300, frame_height=300)
    oak_detection.middle_cam.setInterleaved(False)
    oak_detection.middle_cam.setFps(20)

    # Configure stereo depth
    oak_detection.setDepth()

    # Configure neural network model and init input / output streams
    oak_detection.setModel(path=nn_path)
    oak_detection.labels = labels
    oak_detection.spatial_location_calculator.setWaitForConfigInput(True)

    # Run the loop that call the process function
    oak_detection.run(process=process, middle_cam_queue_size=4, nn_queue_size=4)


# Process function
def process(oak_detection):
    color = (255,0,0)
    frame_width = oak_detection.middle_cam.getPreviewWidth()
    frame_height = oak_detection.middle_cam.getPreviewHeight()
    frame = oak_detection.middle_cam_output_queue.get().getCvFrame()
    tensor = oak_detection.nn_output_queue.get().getLayerFp16("DetectionOutput")
    
    keeped_roi = []
    for i in range(100): # There is 100 detections, not all of them are relevant
        if (tensor[i*7 + 2] >0.5): # 3rd value of each detection is the confidence
            keeped_roi.append(tensor[i*7:i*7+7])

    spatial_calculator_config = dai.SpatialLocationCalculatorConfig()
    for  id, label, confidence, left, top, right, bottom  in  keeped_roi:
        topleft = (int(left*frame_width), int(top*frame_height))
        bottomright = (int(right*frame_width), int(bottom*frame_height))
        cv2.rectangle(frame, topleft, bottomright, color, 2) # ROI
        cv2.putText(frame, oak_detection.labels[int(label)] + f" {int(confidence * 100)}%", (topleft[0] + 10, topleft[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color) # Label and confidence

        # Add ROIs to spatial location calculator config
        spatial_config_data = dai.SpatialLocationCalculatorConfigData()
        spatial_config_data.roi = dai.Rect(dai.Point2f(topleft[0], topleft[1]), dai.Point2f(bottomright[0], bottomright[1]))
        spatial_calculator_config.addROI(spatial_config_data)

    # Put spatial location info inside of the ROI
    if(len(keeped_roi)>0):
        oak_detection.spatial_calculator_input_queue.send(spatial_calculator_config)
        spatial_data = oak_detection.spatial_calculator_output_queue.get().getSpatialLocations()
        for depth_data in spatial_data:
            cv2.putText(frame, f"X: {int(depth_data.spatialCoordinates.x)} mm", (topleft[0] + 10, topleft[1] + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Y: {int(depth_data.spatialCoordinates.y)} mm", (topleft[0] + 10, topleft[1] + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Z: {int(depth_data.spatialCoordinates.z)} mm", (topleft[0] + 10, topleft[1] + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

    cv2.imshow("output", frame)


if __name__ == "__main__":
    main()