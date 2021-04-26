import cv2


def drawROI(frame, topleft, bottomright, label=None, confidence=None, spatialCoordinates=None, color=(0,0,255)):
    # Convert percentage coordinates into pixel coordinates (if necessary)
    if(not isinstance(topleft, (int, int)) and not isinstance(bottomright, (int, int))):
        frame_width, frame_height = frame.shape[:2]
        topleft = (int(topleft[0]*frame_width), int(topleft[1]*frame_height))
        bottomright = (int(bottomright[0]*frame_width), int(bottomright[1]*frame_height))

    # Draw ROI
    cv2.rectangle(frame, topleft, bottomright, color, 2)
    
    # Classify the detection
    if(label is not None):
        if(confidence is not None):
            cv2.putText(frame, label + f" {int(confidence * 100)}%", (topleft[0] + 10, topleft[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        else:
            cv2.putText(frame, label, (topleft[0] + 10, topleft[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

    # Display spatial coordinates
    if(spatialCoordinates is not None):
        cv2.putText(frame, f"X: {int(spatialCoordinates.x)} mm", (topleft[0] + 10, topleft[1] + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Y: {int(spatialCoordinates.y)} mm", (topleft[0] + 10, topleft[1] + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Z: {int(spatialCoordinates.z)} mm", (topleft[0] + 10, topleft[1] + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
    return frame


def displayFPS(frame, fps, color=(255,255,255)):
    cv2.putText(frame, f"fps: {int(fps)}", (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))