from pygame.locals import *
from OpenGL.GLU import *
from OpenGL.GL import *
import numpy as np
import threading
import pygame



class LandmarkCubeVisualizer:
    def __init__(self, window_width, window_height, size, cameras_positions, colors=[]):
        self.__size = size
        self.__cameras_positions = cameras_positions
        self.__window_width = window_width
        self.__window_height = window_height
        self.colors = colors
        
        self.__verticies = ((size, -size, -size), (size, size, -size), (-size, size, -size), (-size, -size, -size),
        (size, -size, size), (size, size, size), (-size, -size, size), (-size, size, size))
        self.__edges = ((0,1), (0,3), (0,4), (2,1), (2,3), (2,7), (6,3), (6,4), (6,7), (5,1), (5,4), (5,7))
        self.__landmarks = []
        self.__centered_landmarks = []


    def __drawCube(self):
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.3, 1.0)
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
            fit = [(self.__size > (landmarks[i][0]-(x_min-self.__cameras_positions[0][0])+(-self.__size-(x_min-(x_min-self.__cameras_positions[0][0])))/2+(self.__size-(x_max-(x_min-self.__cameras_positions[0][0])))/2) > -self.__size) and (self.__size > landmarks[i][1] > -self.__size) and (self.__size > landmarks[i][2] > -self.__size) for i in range(len(landmarks))]

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
                glColor3f(1.0, 1.0, 0.2)
                for x,y,z in self.__centered_landmarks:
                    glVertex3f(x, y, z)
            glEnd()
            glDisable(GL_POINT_SMOOTH)


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

            glRotatef(1, 0, 0, 3)
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            self.__drawCube()
            self.__drawCenter()
            self.__centerLandmarks()
            self.__drawLandmarks()
            pygame.display.flip()
            pygame.time.wait(50)


    def start(self):
        t = threading.Thread(target=self.__run)
        t.start()