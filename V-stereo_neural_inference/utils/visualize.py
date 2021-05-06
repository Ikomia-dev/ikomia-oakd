from pygame.locals import *
from OpenGL.GLU import *
from OpenGL.GL import *
import numpy as np
import threading
import pygame
import math



class LandmarkCubeVisualizer:
    def __init__(self, window_width, window_height, size, cameras_positions, colors=[], pairs=[]):
        self.__size = size
        self.__cameras_positions = cameras_positions
        self.__window_width = window_width
        self.__window_height = window_height
        self.colors = colors
        self.pairs = pairs
        
        scale = 1.05
        self.__verticies = ((size*scale, -size*scale, -size*scale), (size*scale, size*scale, -size*scale), (-size*scale, size*scale, -size*scale), (-size*scale, -size*scale, -size*scale),
        (size*scale, -size*scale, size*scale), (size*scale, size*scale, size*scale), (-size*scale, -size*scale, size*scale), (-size*scale, size*scale, size*scale))
        self.__edges = ((0,1), (0,3), (0,4), (2,1), (2,3), (2,7), (6,3), (6,4), (6,7), (5,1), (5,4), (5,7))
        self.__landmarks = []
        self.__centered_landmarks = []
        self.__roi = ()
        self.__lastPosX = 0


    def __mouseMove(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 4:
            glScaled(1.05, 1.05, 1.05)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 5:
            glScaled(0.95, 0.95, 0.95)

        if event.type == pygame.MOUSEMOTION:
            x, y = event.pos
            dx = x - self.__lastPosX

            mouseState = pygame.mouse.get_pressed()
            if mouseState[0]:
                if(dx != 0):
                    glRotatef(math.sqrt(dx * dx), 0, 0, dx/(dx*dx))

            self.__lastPosX = x


    def __drawCube(self):
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glColor3f(0.4, 0.4, 0.4)
        for edge in self.__edges:
            for vertex in edge:
                glVertex3fv(self.__verticies[vertex])
        glEnd()


    def __drawCenter(self):
        glLineWidth(0.2)
        glBegin(GL_LINES)

        # x
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(self.__size/2, 0, 0)

        # y
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, self.__size/2, 0)

        # z
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, self.__size/2)

        glEnd()


    def setLandmarks(self, landmarks):
        if(len(landmarks)>0):
            x_min = landmarks[0][0]
            x_max = landmarks[0][0]
            for x,y,z in landmarks:
                if(x < x_min):
                    x_min = x
                elif(x > x_max):
                    x_max = x
            fit = [(self.__size > (landmarks[i][0]-(x_min-self.__cameras_positions[0][0])+(-self.__size-(x_min-(x_min-self.__cameras_positions[0][0])))/2+(self.__size-(x_max-(x_min-self.__cameras_positions[0][0])))/2) > -self.__size) and (self.__size > landmarks[i][1] > -self.__size) and (self.__size > landmarks[i][2] > -self.__size) and landmarks[i][0] > 0 for i in range(len(landmarks))]

            if(np.alltrue(fit)):
                self.__landmarks = landmarks
                self.__x_min = x_min
                self.__x_max = x_max


    def __centerLandmarks(self):
        if(len(self.__landmarks)>0):
            self.__centered_landmarks = []
            for i in range(len(self.__landmarks)):
                self.__centered_landmarks.append([self.__landmarks[i][0]-(self.__x_min-self.__cameras_positions[0][0])+(-self.__size-(self.__x_min-(self.__x_min-self.__cameras_positions[0][0])))/2+(self.__size-(self.__x_max-(self.__x_min-self.__cameras_positions[0][0])))/2, self.__landmarks[i][1], self.__landmarks[i][2]])


    def __drawLandmarks(self):
        length = len(self.__centered_landmarks)
        if(length > 0):
            glPointSize(5.0)
            glEnable(GL_POINT_SMOOTH)
            glBegin(GL_POINTS)
            if(length <= len(self.colors)):
                for i in range(length):
                    glColor3f(self.colors[i][0], self.colors[i][1], self.colors[i][2])
                    glVertex3f(self.__centered_landmarks[i][0], self.__centered_landmarks[i][1], self.__centered_landmarks[i][2])
            else:
                glColor3f(1.0, 1.0, 1.0)
                for x,y,z in self.__centered_landmarks:
                    glVertex3f(x, y, z)
            glEnd()
            glDisable(GL_POINT_SMOOTH)


    def __drawPairs(self):
        length = len(self.__centered_landmarks)
        if(length > 0):
            glLineWidth(2.0)
            glBegin(GL_LINES)
            if(length <= len(self.colors)):
                for pair in self.pairs:
                    color = [0.0, 0.0, 0.0]
                    for landmark_index in pair:
                        for i in range(3):
                            color[i] += self.colors[landmark_index][i]/(255*len(pair))
                    glColor3f(color[0], color[1], color[2])
                    for landmark_index in pair:
                        glVertex3fv(self.__centered_landmarks[landmark_index])
            else:
                glColor3f(0.1, 0.8, 0.8)
                for pair in self.pairs:
                    for landmark_index in pair:
                        glVertex3fv(self.__centered_landmarks[landmark_index])
            glEnd()

    
    def __determinedROI(self):
        length = len(self.__centered_landmarks)
        if(length > 0):
            xmin, xmax = self.__centered_landmarks[0][0], self.__centered_landmarks[0][0]
            ymin, ymax = self.__centered_landmarks[0][1], self.__centered_landmarks[0][1]
            zmin, zmax = self.__centered_landmarks[0][2], self.__centered_landmarks[0][2]

            for i in range(1, length):
                xmin = self.__centered_landmarks[i][0] if(self.__centered_landmarks[i][0] < xmin) else xmin
                xmax = self.__centered_landmarks[i][0] if(self.__centered_landmarks[i][0] > xmax) else xmax
                ymin = self.__centered_landmarks[i][1] if(self.__centered_landmarks[i][1] < ymin) else ymin
                ymax = self.__centered_landmarks[i][1] if(self.__centered_landmarks[i][1] > ymax) else ymax
                zmin = self.__centered_landmarks[i][2] if(self.__centered_landmarks[i][2] < zmin) else zmin
                zmax = self.__centered_landmarks[i][2] if(self.__centered_landmarks[i][2] > zmax) else zmax

            xmin, xmax = xmin-self.__size/20, xmax+self.__size/20
            ymin, ymax = ymin-self.__size/20, ymax+self.__size/20
            zmin, zmax = zmin-self.__size/20, zmax+self.__size/20

            self.__roi = ((xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin), (xmin, ymin, zmin),
            (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymin, zmax), (xmin, ymax, zmax))


    def __drawROI(self):
        if(len(self.__roi) == 8):
            glLineWidth(0.5)
            glBegin(GL_LINES)
            glColor3f(0.6, 0.6, 0.15)
            for edge in self.__edges:
                for vertex in edge:
                    glVertex3fv(self.__roi[vertex])
            glEnd()


    def __run(self):
        pygame.init()
        display = (self.__window_width,self.__window_height)
        pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

        gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

        glTranslatef(0.0,0.0, -self.__size*4)
        glRotatef(270, 1, 0, 0)
        glRotatef(270, 0, 0, 1)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    break
                self.__mouseMove(event)

            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            self.__drawCube()
            self.__drawCenter()
            self.__centerLandmarks()
            self.__drawLandmarks()
            self.__drawPairs()
            self.__determinedROI()
            self.__drawROI()
            pygame.display.flip()
            pygame.time.wait(10)


    def start(self):
        t = threading.Thread(target=self.__run)
        t.start()



class LandmarkDepthVisualizer:
    def __init__(self, window_width, window_height, dist, cameras_positions, colors=[], pairs=[]):
        self.dist = dist
        self.__cameras_positions = cameras_positions
        self.__window_width = window_width
        self.__window_height = window_height
        self.colors = colors
        self.pairs = pairs

        self.__landmarks = []
        

    def __drawCenter(self):
        glLineWidth(0.2)
        glBegin(GL_LINES)

        # x
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(1/12, 0, 0)

        # y
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 1/12, 0)

        # z
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 1/12)

        glEnd()


    def setLandmarks(self, landmarks):
        if(len(landmarks)>0):
            self.__x_min = landmarks[0][0]
            self.__x_max = landmarks[0][0]
            

            fit = [landmarks[i][0] > self.__cameras_positions[1][0] for i in range(len(landmarks))]
            if(np.alltrue(fit)):
                self.__landmarks = landmarks
                for x,y,z in landmarks:
                    if(x < self.__x_min):
                        self.__x_min = x
                    elif(x > self.__x_max):
                        self.__x_max = x


    def __drawLandmarks(self):
        length = len(self.__landmarks)
        if(length > 0):
            glPointSize(5.0)
            glEnable(GL_POINT_SMOOTH)
            glBegin(GL_POINTS)
            if(length <= len(self.colors)):
                for i in range(length):
                    glColor3f(self.colors[i][0], self.colors[i][1], self.colors[i][2])
                    glVertex3f(self.__landmarks[i][0], self.__landmarks[i][1], self.__landmarks[i][2])
            else:
                glColor3f(1.0, 1.0, 1.0)
                for x,y,z in self.__landmarks:
                    glVertex3f(x, y, z)
            glEnd()
            glDisable(GL_POINT_SMOOTH)


    def __drawCameras(self):
        if(len(self.__cameras_positions) > 0):
            glPointSize(3.0)
            glBegin(GL_POINTS)
            glColor3f(0.0, 0.5, 1.0)
            for x,y,z in self.__cameras_positions:
                glVertex3f(x, y, z)
            glEnd()


    def __drawVectors(self):
        glLineWidth(0.2)
        glBegin(GL_LINES)
        glColor3f(0.5, 0.5, 0.5)
        for i in range(len(self.__landmarks)):
            for j in range(len(self.__cameras_positions)):
                glVertex3f(self.__cameras_positions[j][0], self.__cameras_positions[j][1], self.__cameras_positions[j][2])
                glVertex3f(self.__landmarks[i][0], self.__landmarks[i][1], self.__landmarks[i][2])
        glEnd()


    def __drawPairs(self):
        length = len(self.__landmarks)
        if(length > 0):
            glLineWidth(1.0)
            glBegin(GL_LINES)
            if(length <= len(self.colors)):
                for pair in self.pairs:
                    color = [0.0, 0.0, 0.0]
                    for landmark_index in pair:
                        for i in range(3):
                            color[i] += self.colors[landmark_index][i]/(255*len(pair))
                    glColor3f(color[0], color[1], color[2])
                    for landmark_index in pair:
                        glVertex3fv(self.__landmarks[landmark_index])
            else:
                glColor3f(0.1, 0.8, 0.8)
                for pair in self.pairs:
                    for landmark_index in pair:
                        glVertex3fv(self.__landmarks[landmark_index])
            glEnd()


    def __run(self):
        pygame.init()
        display = (self.__window_width,self.__window_height)
        pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

        gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

        glTranslatef(-self.dist ,0.0, -self.dist)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    break

            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            self.__drawCenter()
            self.__drawCameras()
            self.__drawLandmarks()
            self.__drawPairs()
            self.__drawVectors()
            pygame.display.flip()
            pygame.time.wait(10)


    def start(self):
        t = threading.Thread(target=self.__run)
        t.start()