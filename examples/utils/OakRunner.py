import depthai as dai
import time
import cv2



class OakRunner:
    def __init__(self):
        self.__pipeline = dai.Pipeline()
        self.__middle_cam = None
        self.__right_cam = None
        self.__left_cam = None
        self.__stereo = None
        self.__spatial_location_calculator = None
        self.__right_cam_manip = None
        self.__left_cam_manip = None

        self.labels = dict()
        self.neural_networks = dict()
        self.input_queues = dict()
        self.output_queues = dict()
        self.left_camera_location = (0.107, -0.038, 0.008)
        self.right_camera_location = (0.109, 0.039, 0.008)
        self.custom_arguments = dict()

        self.__framecount = 0
        self.__starttime = None
        self.__input_queue_arguments = dict()
        self.__output_queue_arguments = dict()
        
        self.__spatial_location_calculator_stream_name = None
        self.__middle_cam_stream_name = None
        self.__right_cam_stream_name = None
        self.__left_cam_stream_name = None


    def __prepareStreamQueue(self, stream_name, queue_size, block_queue, dictionary):
        dictionary[stream_name] = dict()
        dictionary[stream_name]["maxSize"] = queue_size
        dictionary[stream_name]["blocking"] = block_queue

    def setMiddleCamera(self, frame_width, frame_height, stream_name="middle_cam", output_queue_size=1, block_output_queue=False):
        # Needed configuration
        self.__middle_cam = self.__pipeline.create(dai.node.ColorCamera)
        self.__middle_cam.setPreviewSize(frame_width, frame_height)

        # Set output stream
        middle_cam_output_stream = self.__pipeline.create(dai.node.XLinkOut)
        middle_cam_output_stream.setStreamName(stream_name)
        self.__middle_cam.preview.link(middle_cam_output_stream.input)
        self.__middle_cam_stream_name = stream_name
        self.__prepareStreamQueue(stream_name, output_queue_size, block_output_queue, self.__output_queue_arguments)
        


    def setRightCamera(self, frame_width, frame_height, sensor_resolution=dai.MonoCameraProperties.SensorResolution.THE_720_P, stream_name="right_cam", output_queue_size=1, block_output_queue=False):
        # Init right cam
        self.__right_cam = self.__pipeline.create(dai.node.MonoCamera)
        self.__right_cam.setResolution(sensor_resolution)
        self.__right_cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Convert image output into proper sized output
        self.__right_cam_manip = self.__pipeline.create(dai.node.ImageManip)
        self.__right_cam_manip.initialConfig.setResize(frame_width, frame_height)
        self.__right_cam.out.link(self.__right_cam_manip.inputImage)

        # Set output stream
        right_cam_manip_output_stream = self.__pipeline.create(dai.node.XLinkOut)
        right_cam_manip_output_stream.setStreamName(stream_name)
        self.__right_cam_manip.out.link(right_cam_manip_output_stream.input)

        # Prepare streaming queue
        self.__right_cam_stream_name = stream_name
        self.__prepareStreamQueue(stream_name, output_queue_size, block_output_queue, self.__output_queue_arguments)


    def setLeftCamera(self, frame_width, frame_height, sensor_resolution=dai.MonoCameraProperties.SensorResolution.THE_720_P, stream_name="left_cam", output_queue_size=1, block_output_queue=False):
        # Init left cam
        self.__left_cam = self.__pipeline.create(dai.node.MonoCamera)
        self.__left_cam.setResolution(sensor_resolution)
        self.__left_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)

        # Convert image output into proper sized output
        self.__left_cam_manip = self.__pipeline.create(dai.node.ImageManip)
        self.__left_cam_manip.initialConfig.setResize(frame_width, frame_height)
        self.__left_cam.out.link(self.__left_cam_manip.inputImage)
        
        # Set output stream
        left_cam_manip_output_stream = self.__pipeline.create(dai.node.XLinkOut)
        left_cam_manip_output_stream.setStreamName(stream_name)
        self.__left_cam_manip.out.link(left_cam_manip_output_stream.input)

        # Prepare streaming queue
        self.__left_cam_stream_name = stream_name
        self.__prepareStreamQueue(stream_name, output_queue_size, block_output_queue, self.__output_queue_arguments)


    def setMonoDepth(self, sensor_resolution=dai.MonoCameraProperties.SensorResolution.THE_720_P):
        # Warn user
        if((self.__left_cam is not None and self.__right_cam is None and self.__left_cam.getResolution() != sensor_resolution)
         or (self.__left_cam is None and self.__right_cam is not None and sensor_resolution != self.__right_cam.getResolution())
         or (self.__left_cam is not None and self.__right_cam is not None and self.__right_cam.getResolution() != self.__right_cam.getResolution())):
            print("Can't handle mono depth, right and left cameras should have the same sensor resolution, skipped..")
        else:
            # Configure cameras
            if(self.__left_cam is None):
                self.__left_cam = self.__pipeline.create(dai.node.MonoCamera)
                self.__left_cam.setResolution(sensor_resolution)
                self.__left_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
            if(self.__right_cam is None):
                self.__right_cam = self.__pipeline.create(dai.node.MonoCamera)
                self.__right_cam.setResolution(sensor_resolution)
                self.__right_cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

            # Set depth
            self.__stereo = self.__pipeline.create(dai.node.StereoDepth)
            self.__left_cam.out.link(self.__stereo.left)
            self.__right_cam.out.link(self.__stereo.right)


    def __setSpatialLocationCalculator(self, stream_name, slc_input_queue, slc_block_input, slc_output_queue, slc_block_output):
        # Configure spatial location calculator
        self.__spatial_location_calculator = self.__pipeline.create(dai.node.SpatialLocationCalculator)
        self.__stereo.depth.link(self.__spatial_location_calculator.inputDepth)

        # Set spatial location calculator input/output stream
        spatial_calculator_input_stream = self.__pipeline.create(dai.node.XLinkIn)
        spatial_calculator_input_stream.setStreamName(stream_name)
        spatial_calculator_input_stream.out.link(self.__spatial_location_calculator.inputConfig)
        spatial_calculator_output_stream = self.__pipeline.create(dai.node.XLinkOut)
        spatial_calculator_output_stream.setStreamName(stream_name)
        self.__spatial_location_calculator.out.link(spatial_calculator_output_stream.input)

        self.__prepareStreamQueue(stream_name, slc_input_queue, slc_block_input, self.__input_queue_arguments)
        self.__prepareStreamQueue(stream_name, slc_output_queue, slc_block_output, self.__output_queue_arguments)
        self.__spatial_location_calculator_stream_name = stream_name


    def __setModel(self, stream_name, path, input_queue_size, block_input_queue, output_queue_size, block_output_queue):
        # Needed configuration
        self.neural_networks[stream_name].setBlobPath(path)
        self.__prepareStreamQueue(stream_name, input_queue_size, block_input_queue, self.__input_queue_arguments)
        self.__prepareStreamQueue(stream_name, output_queue_size, block_output_queue, self.__output_queue_arguments)

        # Set intput stream
        nn_input_stream = self.__pipeline.create(dai.node.XLinkIn)
        nn_input_stream.setStreamName(stream_name)
        nn_input_stream.out.link(self.neural_networks[stream_name].input)

        # Set output stream
        nn_output_stream = self.__pipeline.create(dai.node.XLinkOut)
        nn_output_stream.setStreamName(stream_name)
        self.neural_networks[stream_name].out.link(nn_output_stream.input)


    def addNeuralNetworkModel(self, stream_name, path, input_queue_size=1, block_input_queue=False, output_queue_size=1, block_output_queue=False, handle_mono_depth=True, slc_stream_name="slc", slc_input_queue=1, slc_block_input=False, slc_output_queue=1, slc_block_output=False):
        self.neural_networks[stream_name] = self.__pipeline.create(dai.node.NeuralNetwork)
        self.__setModel(stream_name, path, input_queue_size, block_input_queue, output_queue_size, block_output_queue)
        if(handle_mono_depth):
            if(self.__stereo is None): # Warn user
                print("To link the mono depth you should configure it (call setMonoDepth), skipped..")
            elif(self.__spatial_location_calculator is None): # If necessary, init the spatial location calculator
                self.__setSpatialLocationCalculator(slc_stream_name, slc_input_queue, slc_block_input, slc_output_queue, slc_block_output)
                print("There was no spatial_location_calculator, it has been set.")


    def addMobileNetDetectionModel(self, stream_name, path, input_queue_size=1, block_input_queue=False, output_queue_size=1, block_output_queue=False, treshold=0.5, handle_mono_depth=True):
        if(handle_mono_depth and self.__stereo is not None):
            self.neural_networks[stream_name] = self.__pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
            self.__stereo.depth.link(self.neural_networks[stream_name].inputDepth)
        else:
            if(handle_mono_depth): # Warn user
                print("To link the mono depth you should configure it (call setMonoDepth), skipped..")
            self.neural_networks[stream_name] = self.__pipeline.create(dai.node.MobileNetDetectionNetwork)
        self.neural_networks[stream_name].setConfidenceThreshold(treshold)
        self.__setModel(stream_name, path, input_queue_size, block_input_queue, output_queue_size, block_output_queue)


    def addYoloDetectionModel(self, stream_name, path, num_classes, coordinate_size, anchors, anchor_masks, input_queue_size=1, block_input_queue=False, output_queue_size=1, block_output_queue=False, treshold=0.5, handle_mono_depth=True):
        if(handle_mono_depth and self.__stereo is not None):
            self.neural_networks[stream_name] = self.__pipeline.create(dai.node.YoloSpatialDetectionNetwork)
            self.__stereo.depth.link(self.neural_networks[stream_name].inputDepth)
        else:
            if(handle_mono_depth): # Warn user
                print("To link the mono depth you should configure it (call setMonoDepth), skipped..")
            self.neural_networks[stream_name] = self.__pipeline.create(dai.node.YoloDetectionNetwork)
        self.neural_networks[stream_name].setConfidenceThreshold(treshold)
        self.neural_networks[stream_name].setNumClasses(num_classes)
        self.neural_networks[stream_name].setCoordinateSize(coordinate_size)
        self.neural_networks[stream_name].setAnchors(anchors)
        self.neural_networks[stream_name].setAnchorMasks(anchor_masks)
        self.__setModel(stream_name, path, input_queue_size, block_input_queue, output_queue_size, block_output_queue)

    
    def getFPS(self):
        if(self.__starttime):
            return self.__framecount / (time.time() - self.__starttime)
        else:
            return 0

    def getPipeline(self):
        return self.__pipeline

    def getMiddleCamera(self):
        return self.__middle_cam

    def getRightCamera(self):
        return self.__right_cam

    def getLeftCamera(self):
        return self.__left_cam

    def getRightCameraManip(self):
        return self.__right_cam_manip

    def getLeftCameraManip(self):
        return self.__left_cam_manip

    def getStereo(self):
        return self.__stereo

    def getSpatialLocationCalculator(self):
        return self.__spatial_location_calculator


    def run(self, process, init=None):
        # Find the connected device (example: OAK-D)
        with dai.Device(self.__pipeline) as device:
            device.startPipeline()

            # Define queues
            if(self.__middle_cam_stream_name is not None):
                self.output_queues[self.__middle_cam_stream_name] = device.getOutputQueue(name=self.__middle_cam_stream_name, maxSize=self.__output_queue_arguments[self.__middle_cam_stream_name]["maxSize"], blocking=self.__output_queue_arguments[self.__middle_cam_stream_name]["blocking"])
            if(self.__right_cam_stream_name is not None):
                self.output_queues[self.__right_cam_stream_name] = device.getOutputQueue(name=self.__right_cam_stream_name, maxSize=self.__output_queue_arguments[self.__right_cam_stream_name]["maxSize"], blocking=self.__output_queue_arguments[self.__right_cam_stream_name]["blocking"])
            if(self.__left_cam_stream_name is not None):
                self.output_queues[self.__left_cam_stream_name] = device.getOutputQueue(name=self.__left_cam_stream_name, maxSize=self.__output_queue_arguments[self.__left_cam_stream_name]["maxSize"], blocking=self.__output_queue_arguments[self.__left_cam_stream_name]["blocking"])
            for name in self.neural_networks.keys():
                self.input_queues[name] = device.getInputQueue(name=name, maxSize=self.__input_queue_arguments[name]["maxSize"], blocking=self.__input_queue_arguments[name]["blocking"])
                self.output_queues[name] = device.getOutputQueue(name=name, maxSize=self.__output_queue_arguments[name]["maxSize"], blocking=self.__output_queue_arguments[name]["blocking"])
            if(self.__spatial_location_calculator_stream_name is not None):
                self.input_queues[self.__spatial_location_calculator_stream_name] = device.getInputQueue(name=self.__spatial_location_calculator_stream_name, maxSize=self.__input_queue_arguments[self.__spatial_location_calculator_stream_name]["maxSize"], blocking=self.__input_queue_arguments[self.__spatial_location_calculator_stream_name]["blocking"])
                self.output_queues[self.__spatial_location_calculator_stream_name] = device.getOutputQueue(name=self.__spatial_location_calculator_stream_name, maxSize=self.__output_queue_arguments[self.__spatial_location_calculator_stream_name]["maxSize"], blocking=self.__output_queue_arguments[self.__spatial_location_calculator_stream_name]["blocking"])

            if(init is not None):
                init(self) # Call the init function

            self.__starttime = time.time()
            # Enter into processing loop
            while True:
                process(self) # Call the process function
                self.__framecount += 1
                if cv2.waitKey(1) == ord('q'):
                    break