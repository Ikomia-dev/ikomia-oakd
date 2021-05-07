from pygame.locals import *
from OpenGL.GLU import *
from OpenGL.GL import *
import numpy as np
import threading
import pygame
import math



def getMinMax(landmarks):
    xmin, xmax = landmarks[0][0], landmarks[0][0]
    ymin, ymax = landmarks[0][1], landmarks[0][1]
    zmin, zmax = landmarks[0][2], landmarks[0][2]

    for i in range(1, len(landmarks)):
        xmin = landmarks[i][0] if(landmarks[i][0] < xmin) else xmin
        xmax = landmarks[i][0] if(landmarks[i][0] > xmax) else xmax
        ymin = landmarks[i][1] if(landmarks[i][1] < ymin) else ymin
        ymax = landmarks[i][1] if(landmarks[i][1] > ymax) else ymax
        zmin = landmarks[i][2] if(landmarks[i][2] < zmin) else zmin
        zmax = landmarks[i][2] if(landmarks[i][2] > zmax) else zmax

    return xmin, xmax, ymin, ymax, zmin, zmax



class LandmarksVisualizer:
    def __init__(self, window_width, window_height, cameras_positions, colors=[], pairs=[]):
        self._cameras_positions = cameras_positions
        self._window_width = window_width
        self._window_height = window_height
        self._landmarks = []
        self._thread = None
        self.colors = colors
        self.pairs = pairs

        self._lastPosX = 0.0
        self._lastPosY = 0.0

        self.x_axis_size = 0.1
        self.y_axis_size = 0.1
        self.z_axis_size = 0.1
        
        self.pair_width = 2.0
        self.axis_width = 0.2
        self.vector_width = 0.2
        
        self.camera_radius = 3.0
        self.landmark_radius = 5.0

        self.camera_color = (0.0, 0.5, 1.0)
        self.vector_color = (0.5, 0.5, 0.5)
        self.pair_default_color = (0.1, 0.8, 0.8)
        self.landmark_default_color = (1.0, 1.0, 1.0)


    def _mouseMove(self, event, allowRotationX=True, allowRotationY=True):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 4:
            glScaled(1.05, 1.05, 1.05)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 5:
            glScaled(0.95, 0.95, 0.95)

        if event.type == pygame.MOUSEMOTION:
            x, y = event.pos
            dx = x - self._lastPosX
            dy = y - self._lastPosY

            mouseState = pygame.mouse.get_pressed()
            if((allowRotationX or allowRotationY) and mouseState[0]):
                if(not allowRotationY): # x rotation
                    if(dx != 0):
                        glRotatef(math.sqrt(dx*dx), 0, 0, 1/dx)
                elif(not allowRotationX): # y rotation
                    if(dy != 0):
                        glRotatef(math.sqrt(dy*dy), 1/dy, 0, 0)
                else:
                    modelView = (GLfloat * 16)()
                    mvm = glGetFloatv(GL_MODELVIEW_MATRIX, modelView)
                    temp = (GLfloat * 3)()
                    temp[0] = modelView[0] * dy + modelView[1] * dx
                    temp[1] = modelView[4] * dy + modelView[5] * dx
                    temp[2] = modelView[8] * dy + modelView[9] * dx
                    norm_xy = math.sqrt(temp[0] * temp[0] + temp[1] * temp[1] + temp[2] * temp[2])
                    glRotatef(math.sqrt(dx * dx + dy * dy), temp[0] / norm_xy, temp[1] / norm_xy, temp[2] / norm_xy)

            self._lastPosX = x
            self._lastPosY = y

    
    def _drawAxis(self):
        glLineWidth(self.axis_width)
        glBegin(GL_LINES)

        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(self.x_axis_size, 0, 0)

        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, self.y_axis_size, 0)

        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, self.z_axis_size)

        glEnd()


    def setLandmarks(self, landmarks):
        if(len(landmarks) > 0):
            self._xmin, self._xmax, self._ymin, self._ymax, self._zmin, self._zmax = getMinMax(landmarks)
            self._landmarks = landmarks


    def _drawLandmarks(self):
        length = len(self._landmarks)
        if(length > 0):
            glPointSize(self.landmark_radius)
            glEnable(GL_POINT_SMOOTH)
            glBegin(GL_POINTS)
            if(length <= len(self.colors)):
                for i in range(length):
                    glColor3f(self.colors[i][0], self.colors[i][1], self.colors[i][2])
                    glVertex3f(self._landmarks[i][0], self._landmarks[i][1], self._landmarks[i][2])
            else:
                glColor3f(self.landmark_default_color[0], self.landmark_default_color[1], self.landmark_default_color[2])
                for x,y,z in self._landmarks:
                    glVertex3f(x, y, z)
            glEnd()
            glDisable(GL_POINT_SMOOTH)


    def _drawPairs(self):
        length = len(self._landmarks)
        if(length > 0):
            glLineWidth(self.pair_width)
            glBegin(GL_LINES)
            if(length <= len(self.colors)):
                for pair in self.pairs:
                    color = [0.0, 0.0, 0.0]
                    for landmark_index in pair:
                        for i in range(3):
                            color[i] += self.colors[landmark_index][i]/(255*len(pair))
                    glColor3f(color[0], color[1], color[2])
                    for landmark_index in pair:
                        glVertex3fv(self._landmarks[landmark_index])
            else:
                glColor3f(self.pair_default_color[0], self.pair_default_color[1], self.pair_default_color[2])
                for pair in self.pairs:
                    for landmark_index in pair:
                        glVertex3fv(self._landmarks[landmark_index])
            glEnd()

    
    def _drawCameras(self):
        if(len(self._cameras_positions) > 0):
            glPointSize(self.camera_radius)
            glBegin(GL_POINTS)
            glColor3f(self.camera_color[0], self.camera_color[1], self.camera_color[2])
            for x,y,z in self._cameras_positions:
                glVertex3f(x, y, z)
            glEnd()


    def _drawVectors(self):
        glLineWidth(self.vector_width)
        glBegin(GL_LINES)
        glColor3f(self.vector_color[0], self.vector_color[1], self.vector_color[2])
        for i in range(len(self._landmarks)):
            for j in range(len(self._cameras_positions)):
                glVertex3f(self._cameras_positions[j][0], self._cameras_positions[j][1], self._cameras_positions[j][2])
                glVertex3f(self._landmarks[i][0], self._landmarks[i][1], self._landmarks[i][2])
        glEnd()


    def _run(self):
        pygame.init()
        display = (self._window_width,self._window_height)
        pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

        gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -4.0)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    break
                self._mouseMove(event)

            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            self._drawAxis()
            self._drawLandmarks()
            self._drawPairs()
            self._drawCameras()
            self._drawVectors()
            pygame.display.flip()
            pygame.time.wait(10)


    def start(self):
        if(self._thread is not None):
            print("Visualizer is already running, skipped..")
        else:
            self._thread = threading.Thread(target=self._run)
            self._thread.start()



class LandmarksCubeVisualizer(LandmarksVisualizer):
    def __init__(self, window_width, window_height, cameras_positions, size=1, colors=[], pairs=[]):
        super().__init__(window_width, window_height, cameras_positions, colors, pairs)
        self.__size = size
        scale = 1.05
        self.__verticies = ((size*scale, -size*scale, -size*scale), (size*scale, size*scale, -size*scale), (-size*scale, size*scale, -size*scale), (-size*scale, -size*scale, -size*scale),
        (size*scale, -size*scale, size*scale), (size*scale, size*scale, size*scale), (-size*scale, -size*scale, size*scale), (-size*scale, size*scale, size*scale))
        self.__edges = ((0,1), (0,3), (0,4), (2,1), (2,3), (2,7), (6,3), (6,4), (6,7), (5,1), (5,4), (5,7))
        self.__roi = ()

        self.cube_width = 3.0
        self.roi_width = 0.5

        self.cube_color = (0.4, 0.4, 0.4)
        self.roi_color = (0.6, 0.6, 0.15)


    def __drawCube(self):
        glLineWidth(self.cube_width)
        glBegin(GL_LINES)
        glColor3f(self.cube_color[0], self.cube_color[1], self.cube_color[2])
        for edge in self.__edges:
            for vertex in edge:
                glVertex3fv(self.__verticies[vertex])
        glEnd()


    def setLandmarks(self, landmarks):
        if(len(landmarks) > 0):
            xmin, xmax, ymin, ymax, zmin, zmax = getMinMax(landmarks)
            fit = [(self.__size > (landmarks[i][0]-(xmin-self._cameras_positions[0][0])+(-self.__size-(xmin-(xmin-self._cameras_positions[0][0])))/2+(self.__size-(xmax-(xmin-self._cameras_positions[0][0])))/2) > -self.__size) and (self.__size > landmarks[i][1] > -self.__size) and (self.__size > landmarks[i][2] > -self.__size) and landmarks[i][0] > 0 for i in range(len(landmarks))]
            if(np.alltrue(fit)):
                self._landmarks = []
                x_translation = -(xmin-self._cameras_positions[0][0])+(-self.__size-(xmin-(xmin-self._cameras_positions[0][0])))/2+(self.__size-(xmax-(xmin-self._cameras_positions[0][0])))/2
                for i in range(len(landmarks)):
                    self._landmarks.append([landmarks[i][0]+x_translation, landmarks[i][1], landmarks[i][2]])
                self._xmin, self._xmax = xmin+x_translation, xmax+x_translation
                self._ymin, self._ymax = ymin, ymax
                self._zmin, self._zmax = zmin, zmax


    def __drawROI(self):
        if(len(self._landmarks) > 0):
            margin = self.__size/20
            self.__roi = ((self._xmax+margin, self._ymin-margin, self._zmin-margin), (self._xmax+margin, self._ymax+margin, self._zmin-margin), (self._xmin-margin, self._ymax+margin, self._zmin-margin), (self._xmin-margin, self._ymin-margin, self._zmin-margin),
            (self._xmax+margin, self._ymin-margin, self._zmax+margin), (self._xmax+margin, self._ymax+margin, self._zmax+margin), (self._xmin-margin, self._ymin-margin, self._zmax+margin), (self._xmin-margin, self._ymax+margin, self._zmax+margin))
            
            glLineWidth(self.roi_width)
            glBegin(GL_LINES)
            glColor3f(self.roi_color[0], self.roi_color[1], self.roi_color[2])
            for edge in self.__edges:
                for vertex in edge:
                    glVertex3fv(self.__roi[vertex])
            glEnd()


    def _run(self):
        pygame.init()
        display = (self._window_width,self._window_height)
        pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

        gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

        glTranslatef(0.0, 0.0, -self.__size*4)
        glRotatef(270, 1, 0, 0)
        glRotatef(270, 0, 0, 1)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    break
                self._mouseMove(event, allowRotationY=False)

            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            self.__drawCube()
            self._drawAxis()
            self._drawLandmarks()
            self._drawPairs()
            self.__drawROI()
            pygame.display.flip()
            pygame.time.wait(10)



class LandmarksDepthVisualizer(LandmarksVisualizer):
    def __init__(self, window_width, window_height, cameras_positions, colors=[], pairs=[]):
        super().__init__(window_width, window_height, cameras_positions, colors, pairs)


    def setLandmarks(self, landmarks):
        if(len(landmarks)>0):
            xmin, xmax, ymin, ymax, zmin, zmax = getMinMax(landmarks)
            if(xmin>0):
                self._landmarks = landmarks
                self._xmin, self._xmax = xmin, xmax
                self._ymin, self._ymax = ymin, ymax
                self._zmin, self._zmax = zmin, zmax


    def _run(self):
        pygame.init()
        display = (self._window_width,self._window_height)
        pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

        gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
        glTranslatef(-0.25 ,0.0, -1)
        glRotatef(11, 0, 1, 0)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    break
                self._mouseMove(event, allowRotationX=False)

            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            self._drawAxis()
            self._drawCameras()
            self._drawLandmarks()
            self._drawPairs()
            self._drawVectors()
            pygame.display.flip()
            pygame.time.wait(10)