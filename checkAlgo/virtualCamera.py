import time

import numpy.linalg
import vtk
import numpy as np
from scipy.spatial.transform import Rotation
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkPlaneSource
from vtkmodules.vtkIOImage import vtkImageReader2Factory, vtkPNGWriter
from vtkmodules.vtkRenderingCore import vtkTexture, vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow, \
    vtkWindowToImageFilter



class PlaneRenderer():
    def __init__(self, windowWidth, windowHeight, cameraMatrix, imagePath, bkgColor=None):
        if bkgColor is None:
            bkgColor = [100, 0, 0, 255]
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.cameraMatrix = cameraMatrix
        self.imagePath = imagePath
        self.bkgColor = bkgColor

        self.f = np.array([cameraMatrix[0, 0], cameraMatrix[1, 1]])
        self.c = cameraMatrix[:2, 2]

        self.init_vtk()

    def renderPlane(self, planePosition: np.array, planeRotation: Rotation, planeLength: float, saveTo: str):
        self.plane = vtkPlaneSource()
        self.plane.SetOrigin(-planeLength*0.5, -planeLength*0.5, 0.0)
        self.plane.SetPoint1(planeLength*0.5, -planeLength*0.5, 0.0)
        self.plane.SetPoint2(-planeLength*0.5, planeLength*0.5, 0.0)
        self.plane.SetCenter(planePosition[0], planePosition[1], planePosition[2])
        rotVec = planeRotation.as_rotvec(degrees=True)
        self.plane.Rotate(numpy.linalg.norm(rotVec), (rotVec[0], rotVec[1], rotVec[2]))

        planeMapper = vtkPolyDataMapper()
        planeMapper.SetInputConnection(self.plane.GetOutputPort())

        planeActor = vtkActor()
        planeActor.SetMapper(planeMapper)
        planeActor.SetTexture(self.textureMap)
        planeActor.GetProperty().LightingOff()
        self.renderer.AddActor(planeActor)

        w2if = vtkWindowToImageFilter()
        w2if.SetInput(self.renWin)
        w2if.Update()

        writer = vtkPNGWriter()
        writer.SetFileName(saveTo)
        writer.SetInputData(w2if.GetOutput())
        writer.Write()
        self.renderer.RemoveActor(planeActor)

    def init_vtk(self):
        self.colors = vtkNamedColors()
        self.colors.SetColor('BkgColor', self.bkgColor)

        readerFactory = vtkImageReader2Factory()
        textureFile = readerFactory.CreateImageReader2(self.imagePath)
        textureFile.SetFileName(self.imagePath)
        textureFile.Update()

        self.textureMap = vtkTexture()
        self.textureMap.SetInputConnection(textureFile.GetOutputPort())
        self.textureMap.InterpolateOff()

        self.renderer = vtkRenderer()
        self.renWin = vtkRenderWindow()
        self.renWin.AddRenderer(self.renderer)
        self.renWin.SetShowWindow(False)

        self.renderer.SetBackground(self.colors.GetColor3d('BkgColor'))
        self.renWin.SetSize(self.windowWidth, self.windowHeight)

        self.renderer.ResetCamera()
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(0, 0, 0)
        cam.SetFocalPoint(0, 0, 1)
        cam.SetViewUp(0, -1, 0)
        wcx = -2.0 * (self.c[0] - self.windowWidth / 2.0) / self.windowWidth
        wcy = 2.0 * (self.c[1] - self.windowHeight / 2.0) / self.windowHeight
        cam.SetWindowCenter(wcx, wcy)
        angle = 180 / np.pi * 2.0 * np.arctan2(self.windowHeight / 2.0, self.f[1])
        cam.SetViewAngle(angle)
        m = np.eye(4)
        aspect = self.f[1] / self.f[0]
        m[0, 0] = 1.0 / aspect
        t = vtk.vtkTransform()
        t.SetMatrix(m.flatten())
        cam.SetUserTransform(t)
        self.renderer.ResetCameraClippingRange()
