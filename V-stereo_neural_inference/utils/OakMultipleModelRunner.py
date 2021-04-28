import depthai as dai
import time
import cv2


middle_cam_stream_name = "middle_cam"
left_cam_manip_stream_name = "left_cam_manip"
right_cam_manip_stream_name = "right_cam_manip"
spatial_calculator_input_stream_name = "spatial_config"
spatial_calculator_output_stream_name = "spatial_data"


class OakMultipleModelRunner:
    def __init__(self):
        self.__pipeline = dai.Pipeline()
        self.middle_cam = None
        self.left_cam = None
        self.right_cam = None
        self.left_cam_manip = None
        self.right_cam_manip = None

        self.spatial_location_calculator = None
        self.stereo = None

        self.neural_networks = dict()
        self.nn_output_queue = dict()
        self.labels = dict()

        self.__nn_output_queue_parameters = dict()
        self.__starttime = None
        self.__framecount = 0


    def setMiddleCamera(self, frame_width, frame_height):
        # Needed configuration
        self.middle_cam = self.__pipeline.createColorCamera()
        self.middle_cam.setPreviewSize(frame_width, frame_height)

        # Set output stream
        middle_cam_output_stream = self.__pipeline.createXLinkOut()
        middle_cam_output_stream.setStreamName(middle_cam_stream_name)
        self.middle_cam.preview.link(middle_cam_output_stream.input)


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
        left_cam_manip_output_stream = self.__pipeline.createXLinkOut()
        left_cam_manip_output_stream.setStreamName(left_cam_manip_stream_name)
        self.left_cam_manip.out.link(left_cam_manip_output_stream.input)


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
        right_cam_manip_output_stream = self.__pipeline.createXLinkOut()
        right_cam_manip_output_stream.setStreamName(right_cam_manip_stream_name)
        self.right_cam_manip.out.link(right_cam_manip_output_stream.input)


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
        spatial_calculator_input_stream = self.__pipeline.createXLinkIn()
        spatial_calculator_input_stream.setStreamName(spatial_calculator_input_stream_name)
        spatial_calculator_input_stream.out.link(self.spatial_location_calculator.inputConfig)
        spatial_calculator_output_stream = self.__pipeline.createXLinkOut()
        spatial_calculator_output_stream.setStreamName(spatial_calculator_output_stream_name)
        self.spatial_location_calculator.out.link(spatial_calculator_output_stream.input)


    def __setModel(self, name, path):
        # Needed configuration
        self.neural_networks[name].setBlobPath(path)

        # Set output stream
        nn_output_stream = self.__pipeline.createXLinkOut()
        nn_output_stream.setStreamName(name)
        self.neural_networks[name].out.link(nn_output_stream.input)


    def addNeuralNetworkModel(self, name, path, output_queue_size=1, block_output_queue=False, handle_depth=True):
        self.neural_networks[name] = self.__pipeline.createNeuralNetwork()
        self.__nn_output_queue_parameters[name] = [output_queue_size, block_output_queue]
        self.__setModel(name, path)
        if(handle_depth):
            if(self.stereo is None): # Warn user
                print("To link the depth you should configure it (call setDepth), skipped..")
            elif(self.spatial_location_calculator is None): # If necessary, init the spatial location calculator
                self.__setSpatialLocationCalculator()
                print("There was no spatial_location_calculator, it has been set.")


    def addMobileNetDetectionModel(self, name, path, output_queue_size=1, block_output_queue=False, treshold=0.5, handle_depth=True):
        self.__nn_output_queue_parameters[name] = [output_queue_size, block_output_queue]
        if(handle_depth and self.stereo is not None):
            self.neural_networks[name] = self.__pipeline.createMobileNetSpatialDetectionNetwork()
            self.stereo.depth.link(self.neural_networks[name].inputDepth)
        else:
            if(handle_depth): # Warn user
                print("To link the depth you should configure it (call setDepth), skipped..")
            self.neural_networks[name] = self.__pipeline.createMobileNetDetectionNetwork()
        self.neural_networks[name].setConfidenceThreshold(treshold)
        self.__setModel(name, path)


    def addYoloDetectionModel(self, name, path, num_classes, coordinate_size, anchors, anchor_masks, output_queue_size=1, block_output_queue=False, treshold=0.5, handle_depth=True):
        self.__nn_output_queue_parameters[name] = [output_queue_size, block_output_queue]
        if(handle_depth and self.stereo is not None):
            self.neural_networks[name] = self.__pipeline.createYoloSpatialDetectionNetwork()
            self.stereo.depth.link(self.neural_networks[name].inputDepth)
        else:
            if(handle_depth): # Warn user
                print("To link the depth you should configure it (call setDepth), skipped..")
            self.neural_networks[name] = self.__pipeline.createYoloDetectionNetwork()
        self.neural_networks[name].setConfidenceThreshold(treshold)
        self.neural_networks[name].setNumClasses(num_classes)
        self.neural_networks[name].setCoordinateSize(coordinate_size)
        self.neural_networks[name].setAnchors(anchors)
        self.neural_networks[name].setAnchorMasks(anchor_masks)
        self.__setModel(name, path)

    
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
            for name in self.neural_networks.keys():
                self.nn_output_queue[name] = device.getOutputQueue(name=name, maxSize=self.__nn_output_queue_parameters[name][0], blocking=self.__nn_output_queue_parameters[name][1])
            if(self.spatial_location_calculator is not None):
                self.spatial_calculator_input_queue = device.getInputQueue(name=spatial_calculator_input_stream_name, maxSize=spatial_calculator_input_queue_size, blocking=block_spatial_calculator_input_queue)
                self.spatial_calculator_output_queue = device.getOutputQueue(name=spatial_calculator_output_stream_name, maxSize=spatial_calculator_output_queue_size, blocking=block_spatial_calculator_output_queue)

            self.__starttime = time.time()
            # Enter into processing loop
            while True:
                process(self) # Call the process function
                self.__framecount += 1
                if cv2.waitKey(1) == ord('q'):
                    break