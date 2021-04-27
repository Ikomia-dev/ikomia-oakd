import depthai as dai
import time
import cv2


middle_cam_stream_name = "middle_cam"
left_cam_manip_stream_name = "left_cam_manip"
right_cam_manip_stream_name = "right_cam_manip"
nn_stream_name = "nn"
spatial_calculator_input_stream_name = "spatial_config"
spatial_calculator_output_stream_name = "spatial_data"


class OakSingleModelRunner:
    def __init__(self):
        self.__isAnyNeuralNetworkModel = False
        self.__pipeline = dai.Pipeline()
        self.middle_cam = None
        self.left_cam = None
        self.right_cam = None
        self.stereo = None
        self.nn = None
        self.labels = []
        self.__starttime = None
        self.__framecount = 0

        self.__left_cam_output_stream = None
        self.__right_cam_output_stream = None
        self.left_cam_manip = None
        self.right_cam_manip = None


    def setMiddleCamera(self, frame_width, frame_height):
        # Needed configuration
        self.middle_cam = self.__pipeline.createColorCamera()
        self.middle_cam.setPreviewSize(frame_width, frame_height)

        # Set output stream
        self.__middle_cam_output_stream = self.__pipeline.createXLinkOut()
        self.__middle_cam_output_stream.setStreamName(middle_cam_stream_name)
        self.middle_cam.preview.link(self.__middle_cam_output_stream.input)


    def setLeftCamera(self, frame_width, frame_height, sensor_resolution=dai.MonoCameraProperties.SensorResolution.THE_400_P):
        # Init left cam
        self.left_cam = self.__pipeline.createMonoCamera()
        self.left_cam.setResolution(sensor_resolution)
        self.left_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)

        # Convert image output into proper sized output
        self.left_cam_manip = self.__pipeline.createImageManip()
        self.left_cam_manip.initialConfig.setResize(frame_width, frame_height)
        self.left_cam.out.link(self.left_cam_manip.inputImage)
        
        # Set output stream
        self.__left_cam_manip_output_stream = self.__pipeline.createXLinkOut()
        self.__left_cam_manip_output_stream.setStreamName(left_cam_manip_stream_name)
        self.left_cam_manip.out.link(self.__left_cam_manip_output_stream.input)


    def setRightCamera(self, frame_width, frame_height, sensor_resolution=dai.MonoCameraProperties.SensorResolution.THE_400_P):
        # Init right cam
        self.right_cam = self.__pipeline.createMonoCamera()
        self.right_cam.setResolution(sensor_resolution)
        self.right_cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Convert image output into proper sized output
        self.right_cam_manip = self.__pipeline.createImageManip()
        self.right_cam_manip.initialConfig.setResize(frame_width, frame_height)
        self.right_cam.out.link(self.right_cam_manip.inputImage)

        # Set output stream
        self.__right_cam_manip_output_stream = self.__pipeline.createXLinkOut()
        self.__right_cam_manip_output_stream.setStreamName(right_cam_manip_stream_name)
        self.right_cam_manip.out.link(self.__right_cam_manip_output_stream.input)


    def setDepth(self, sensor_resolution=dai.MonoCameraProperties.SensorResolution.THE_400_P):
        # Warn user
        if((self.left_cam is not None and self.right_cam is None and self.left_cam.getResolution() != sensor_resolution)
         or (self.left_cam is None and self.right_cam is not None and sensor_resolution != self.right_cam.getResolution())
         or (self.left_cam is not None and self.right_cam is not None and self.right_cam.getResolution() != self.right_cam.getResolution())):
            print("Can't handle depth, right and left cameras should have the same sensor resolution, skipped..")
        else:
            # Configure cameras
            if(self.left_cam is None):
                self.left_cam = self.__pipeline.createMonoCamera()
                self.left_cam.setResolution(sensor_resolution)
                self.left_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
            if(self.right_cam is None):
                self.right_cam = self.__pipeline.createMonoCamera()
                self.right_cam.setResolution(sensor_resolution)
                self.right_cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

            # Set depth
            self.stereo = self.__pipeline.createStereoDepth()
            self.left_cam.out.link(self.stereo.left)
            self.right_cam.out.link(self.stereo.right)


    def __setSpatialLocationCalculator(self):
        # Configure spatial location calculator
        self.spatial_location_calculator = self.__pipeline.createSpatialLocationCalculator()
        self.stereo.depth.link(self.spatial_location_calculator.inputDepth)

        # Set spatial location calculator input/output stream
        self.__spatial_calculator_input_stream = self.__pipeline.createXLinkIn()
        self.__spatial_calculator_input_stream.setStreamName(spatial_calculator_input_stream_name)
        self.__spatial_calculator_input_stream.out.link(self.spatial_location_calculator.inputConfig)
        self.__spatial_calculator_output_stream = self.__pipeline.createXLinkOut()
        self.__spatial_calculator_output_stream.setStreamName(spatial_calculator_output_stream_name)
        self.spatial_location_calculator.out.link(self.__spatial_calculator_output_stream.input)


    def __setModel(self, path, link_middle_cam):
        # Warn user
        if(link_middle_cam and self.middle_cam is None):
            print("To link the middle camera you should configure it (call setMiddleCamera), skipped..")

        # Needed configuration
        self.nn.setBlobPath(path)

        # Link middle camera to neural network input
        if(link_middle_cam and self.middle_cam is not None):
            self.middle_cam.preview.link(self.nn.input)
            self.nn.passthrough.link(self.__middle_cam_output_stream.input)

        # Set output stream
        self.__nn_output_stream = self.__pipeline.createXLinkOut()
        self.__nn_output_stream.setStreamName(nn_stream_name)
        self.nn.out.link(self.__nn_output_stream.input)


    def setNeuralNetworkModel(self, path, link_middle_cam=True, link_depth=True):
        self.nn = self.__pipeline.createNeuralNetwork()
        self.__setModel(path, link_middle_cam)
        self.__isAnyNeuralNetworkModel = True
        if(link_depth):
            if(self.stereo is None): # Warn user
                print("To link the depth you should configure it (call setDepth), skipped..")
            else: # Init the spatial location calculator
                self.__setSpatialLocationCalculator()


    def setMobileNetDetectionModel(self, path, treshold=0.5, link_middle_cam=True, link_depth=True):
        if(link_depth):
            if(self.stereo is None): # Warn user
                print("To link the depth you should configure it (call setDepth), skipped..")
                self.nn = self.__pipeline.createMobileNetDetectionNetwork()
            else: # Handle depth
                self.nn = self.__pipeline.createMobileNetSpatialDetectionNetwork()
                self.stereo.depth.link(self.nn.inputDepth)
        self.nn.setConfidenceThreshold(treshold)
        self.__setModel(path, link_middle_cam)


    def setYoloDetectionModel(self, path, num_classes, coordinate_size, anchors, anchor_masks, treshold=0.5, link_middle_cam=True, link_depth=True):
        if(link_depth):
            if(self.stereo is None): # Warn user
                print("To link the depth you should configure it (call setDepth), skipped..")
                self.nn = self.__pipeline.createYoloDetectionNetwork()
            else: # Handle depth
                self.nn = self.__pipeline.createYoloSpatialDetectionNetwork()
                self.stereo.depth.link(self.nn.inputDepth)
        self.nn.setConfidenceThreshold(treshold)
        self.nn.setNumClasses(num_classes)
        self.nn.setCoordinateSize(coordinate_size)
        self.nn.setAnchors(anchors)
        self.nn.setAnchorMasks(anchor_masks)
        self.__setModel(path, link_middle_cam)

    
    def getFPS(self):
        if(self.__starttime):
            return self.__framecount / (time.time() - self.__starttime)
        else:
            return 0


    def run(self, process, middle_cam_queue_size=1, block_middle_cam_queue=False, left_cam_queue_size=1, block_left_cam_queue=False, right_cam_queue_size=1, block_right_cam_queue=False, nn_queue_size=1, block_nn_queue=False, spatial_calculator_input_queue_size=1, block_spatial_calculator_input_queue=False, spatial_calculator_output_queue_size=1, block_spatial_calculator_output_queue=False):
        # Find the connected device (example: OAK-D)
        with dai.Device(self.__pipeline) as device:
            device.startPipeline()

            # Define queues
            if(self.middle_cam is not None):
                self.middle_cam_output_queue = device.getOutputQueue(name=middle_cam_stream_name, maxSize=middle_cam_queue_size, blocking=block_middle_cam_queue)
            if(self.left_cam_manip is not None):
                self.left_cam_output_queue = device.getOutputQueue(name=left_cam_manip_stream_name, maxSize=left_cam_queue_size, blocking=block_left_cam_queue)
            if(self.right_cam_manip is not None):
                self.right_cam_output_queue = device.getOutputQueue(name=right_cam_manip_stream_name, maxSize=right_cam_queue_size, blocking=block_right_cam_queue)
            if(self.nn is not None):
                self.nn_output_queue = device.getOutputQueue(name=nn_stream_name, maxSize=nn_queue_size, blocking=block_nn_queue)
            if(self.__isAnyNeuralNetworkModel and self.stereo is not None):
                self.spatial_calculator_input_queue = device.getInputQueue(name=spatial_calculator_input_stream_name, maxSize=spatial_calculator_input_queue_size, blocking=block_spatial_calculator_input_queue)
                self.spatial_calculator_output_queue = device.getOutputQueue(name=spatial_calculator_output_stream_name, maxSize=spatial_calculator_output_queue_size, blocking=block_spatial_calculator_output_queue)

            self.__starttime = time.time()
            # Enter into processing loop
            while True:
                process(self) # Call the process function
                self.__framecount += 1
                if cv2.waitKey(1) == ord('q'):
                    break