from pathlib import Path
import depthai as dai
import cv2

# Importing from parent folder is harder
import importlib.util
import inspect
spec = importlib.util.spec_from_file_location("OakSingleModelRunner", str(Path(__file__).parent) + "/../OakSingleModelRunner.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
OakSingleModelRunner = inspect.getmembers(module)[0][1]


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
    runner.setMobileNetDetectionModel(path=nn_path)
    runner.labels = labels

    # Run the loop that call the process function
    runner.run(process=process, middle_cam_queue_size=4, nn_queue_size=4)


# Process function
def process(runner):
    color = (255,0,0)
    frame_width = runner.middle_cam.getPreviewWidth()
    frame_height = runner.middle_cam.getPreviewHeight()
    frame = runner.middle_cam_output_queue.get().getCvFrame()

    detections = runner.nn_output_queue.get().detections
    for detection in detections:
        topleft = (int(detection.xmin*frame_width), int(detection.ymin*frame_height))
        bottomright = (int(detection.xmax*frame_width), int(detection.ymax*frame_height))
        cv2.rectangle(frame, topleft, bottomright, color, 2) # ROI
        cv2.putText(frame, runner.labels[detection.label] + f" {int(detection.confidence * 100)}%", (topleft[0] + 10, topleft[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color) # Label and confidence
        cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (topleft[0] + 10, topleft[1] + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (topleft[0] + 10, topleft[1] + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (topleft[0] + 10, topleft[1] + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
    
    cv2.imshow("output", frame)


if __name__ == "__main__":
    main()