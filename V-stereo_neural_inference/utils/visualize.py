from OpenGL.GL import glBegin, glColor3f, glEnd, glPointSize, GL_POINTS, glVertex3f, glTranslatef, glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT
from OpenGL.GLU import gluPerspective
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE
import pygame



def initOpenGL(width, height):
    pygame.init()

    display = (width, height)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL, RESIZABLE)

    gluPerspective(45, (1.0 * display[0] / display[1]), 0.1, 50.0)
    glTranslatef(-5.0, 0.0, -20)


def startOpenGL(landmarks):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    visualizeLandmarks(landmarks)
    pygame.display.flip()
    pygame.time.wait(1)


def visualizeLandmarks(landmarks):
    glPointSize(9.0)
    glBegin(GL_POINTS)
    glColor3f(1.0, 0.0, 0.0)
    for i in range(len(landmarks)):
        glVertex3f(landmarks[i][0], landmarks[i][1], landmarks[i][2])
    glEnd()