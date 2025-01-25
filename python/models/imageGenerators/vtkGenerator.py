import numpy.linalg
import vtk
import numpy as np
from scipy.spatial.transform import Rotation
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkPlaneSource
from vtkmodules.vtkIOImage import vtkImageReader2Factory, vtkPNGWriter
from vtkmodules.vtkRenderingCore import vtkTexture, vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow, \
    vtkWindowToImageFilter

from python.models.imageGenerators.imageGenerator import ImageGenerator


class VTKGenerator(ImageGenerator):
    cameraRotation: Rotation
    cameraTranslation: np.array

    def __init__(self, imageWidth, imageHeight, imagePath, cameraMatrix, planeWidth: float, planeHeight: float, bkgColor=None, cameraRotation: Rotation=None, cameraTranslation: np.array=None):
        super().__init__()
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.planeImagePath = imagePath
        self.cameraMatrix = cameraMatrix
        self.planeWidth = planeWidth
        self.planeHeight = planeHeight

        if bkgColor is None:
            bkgColor = [154, 188, 255, 255]
        self.bkgColor = bkgColor

        if cameraRotation is None:
            cameraRotation = Rotation.from_rotvec([0, 0, 0], degrees=True)
        self.cameraRotation = cameraRotation

        if cameraTranslation is None:
            cameraTranslation = np.zeros((3,))
        self.cameraTranslation = cameraTranslation

        self.f = np.array([cameraMatrix[0, 0], cameraMatrix[1, 1]])
        self.c = cameraMatrix[:2, 2]

        self.init_vtk()

    def makeImageWithPlane(self, planeTranslation: np.array, planeRotation: Rotation, savePath: str):
        self.plane = vtkPlaneSource()
        self.plane.SetOrigin(-self.planeWidth * 0.5, -self.planeHeight * 0.5, 0.0)
        self.plane.SetPoint1(self.planeWidth * 0.5, -self.planeHeight * 0.5, 0.0)
        self.plane.SetPoint2(-self.planeWidth * 0.5, self.planeHeight * 0.5, 0.0)
        self.plane.SetCenter(planeTranslation[0], planeTranslation[1], planeTranslation[2])
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
        writer.SetFileName(savePath)
        writer.SetInputData(w2if.GetOutput())
        writer.Write()
        self.renderer.RemoveActor(planeActor)

    def init_vtk(self):
        self.colors = vtkNamedColors()
        self.colors.SetColor('BkgColor', self.bkgColor)

        readerFactory = vtkImageReader2Factory()
        textureFile = readerFactory.CreateImageReader2(self.planeImagePath)
        textureFile.SetFileName(self.planeImagePath)
        textureFile.Update()

        self.textureMap = vtkTexture()
        self.textureMap.SetInputConnection(textureFile.GetOutputPort())
        self.textureMap.InterpolateOff()

        self.renderer = vtkRenderer()
        self.renWin = vtkRenderWindow()
        self.renWin.AddRenderer(self.renderer)
        self.renWin.SetShowWindow(False)

        self.renderer.SetBackground(self.colors.GetColor3d('BkgColor'))
        self.renWin.SetSize(self.imageWidth, self.imageHeight)

        self.renderer.ResetCamera()
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(self.cameraTranslation[0], self.cameraTranslation[1], self.cameraTranslation[2])
        focalPoint = self.cameraRotation.apply([0, 0, 1])
        cam.SetFocalPoint(self.cameraTranslation[0] + focalPoint[0], self.cameraTranslation[1] + focalPoint[1],
                          self.cameraTranslation[2] + focalPoint[2])
        viewUp = self.cameraRotation.apply([0, -1, 0])
        cam.SetViewUp(self.cameraTranslation[0] + viewUp[0], self.cameraTranslation[1] + viewUp[1],
                      self.cameraTranslation[2] + viewUp[2])
        wcx = -2.0 * (self.c[0] - self.imageWidth / 2.0) / self.imageWidth
        wcy = 2.0 * (self.c[1] - self.imageWidth / 2.0) / self.imageHeight
        cam.SetWindowCenter(wcx, wcy)
        angle = 180 / np.pi * 2.0 * np.arctan2(self.imageHeight / 2.0, self.f[1])
        cam.SetViewAngle(angle)
        m = np.eye(4)
        aspect = self.f[1] / self.f[0]
        m[0, 0] = 1.0 / aspect
        t = vtk.vtkTransform()
        t.SetMatrix(m.flatten())
        cam.SetUserTransform(t)
        self.renderer.ResetCameraClippingRange()
