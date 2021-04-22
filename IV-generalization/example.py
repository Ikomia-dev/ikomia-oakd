from OakSingleModelRunner import OakSingleModelRunner
from pathlib import Path
import cv2


def main():
    # Set few parameters
    nn_path = str(Path(__file__).parent) + "/../models/coronamask.blob"
    labelMap = ["background", "no mask", "mask", "no mask"]

    # Init pipeline
    oak_detection = OakSingleModelRunner() 

    # Configure middle camera and init output streams
    oak_detection.setMiddleCamera(frame_width=300, frame_height=300, fps_limit=20)

    # Configure neural network model and init input / output streams
    oak_detection.setModel(path=nn_path, labels=labelMap)

    # Run the loop that call the process function
    oak_detection.run(process=process, middle_cam_queue_size=4, nn_queue_size=4)


# Process function
def process(oak_detection):
    color = (255,0,0)
    frame = oak_detection.middle_cam_output_queue.get().getCvFrame()
    tensor = oak_detection.nn_output_queue.get().getLayerFp16("DetectionOutput")
    
    keeped_roi = []
    for i in range(100): # There is 100 detections, not all of them are relevant
        if (tensor[i*7 + 2] >0.5): # 3rd value of each detection is the confidence
            keeped_roi.append(tensor[i*7:i*7+7])

    for  id, label, confidence, left, top, right, bottom  in  keeped_roi:
        topleft = (int(left*oak_detection.frame_width), int(top*oak_detection.frame_height))
        bottomright = (int(right*oak_detection.frame_width), int(bottom*oak_detection.frame_height))
        cv2.rectangle(frame, topleft, bottomright, color, 2) # ROI
        cv2.putText(frame, oak_detection.labels[int(label)] + f" {int(confidence * 100)}%", (topleft[0] + 10, topleft[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color) # Label and confidence

    cv2.imshow("output", frame)


if __name__ == "__main__":
    main()