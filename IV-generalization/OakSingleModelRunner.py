import depthai as dai
import time
import cv2


middle_cam_stream_name = "middle_cam"
nn_stream_name = "nn"


class OakSingleModelRunner:
    def __init__(self):
        self.frame_width = None
        self.frame_height = None
        self.labels = None

        self.middle_cam_output_queue = None
        self.nn_output_queue = None

        self.__pipeline = dai.Pipeline()
        self.__middle_cam = None
        self.__nn = None
        

    def setMiddleCamera(self, frame_width, frame_height, fps_limit=None):
        # Needed configuration
        self.__middle_cam = self.__pipeline.createColorCamera()
        self.__middle_cam.setPreviewSize(frame_width, frame_height)
        self.__middle_cam.setInterleaved(False)
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Optional configuration
        if(fps_limit is not None):
            self.__middle_cam.setFps(fps_limit)
        # ... other optional parameters

        # Set output stream
        self.__middle_cam_output_stream = self.__pipeline.createXLinkOut()
        self.__middle_cam_output_stream.setStreamName(middle_cam_stream_name)
        self.__middle_cam.preview.link(self.__middle_cam_output_stream.input)


    def setModel(self, path, labels=None):
        # Needed configuration
        self.__nn = self.__pipeline.createNeuralNetwork()
        self.__nn.setBlobPath(path)

        # Optional configuration
        if(labels is not None):
            self.labels = labels
        # ... other optional parameters

        # Link middle camera to neural network input
        if(self.__middle_cam is not None):
            self.__middle_cam.preview.link(self.__nn.input)
            self.__nn.passthrough.link(self.__middle_cam_output_stream.input)

        # Set output stream
        self.__nn_output_stream = self.__pipeline.createXLinkOut()
        self.__nn_output_stream.setStreamName(nn_stream_name)
        self.__nn.out.link(self.__nn_output_stream.input)
        

    def run(self, process, middle_cam_queue_size=1, nn_queue_size=1):
        # Find the connected device (example: OAK-D)
        with dai.Device(self.__pipeline) as device:
            device.startPipeline()

            # Define queues
            if(self.__middle_cam is not None):
                self.middle_cam_output_queue = device.getOutputQueue(name=middle_cam_stream_name, maxSize=middle_cam_queue_size, blocking=False)
            if(self.__nn is not None):
                self.nn_output_queue = device.getOutputQueue(name=nn_stream_name, maxSize=nn_queue_size, blocking=False)

            # Enter into processing loop
            while True:
                process(self) # Call the process function
                if cv2.waitKey(1) == ord('q'):
                    break