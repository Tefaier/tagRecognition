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
    camera_rotation: Rotation
    camera_translation: np.array

    def __init__(self, image_width, image_height, image_path, cameraMatrix, plane_width: float, plane_height: float, bkg_color=None, camera_rotation: Rotation=None, camera_translation: np.array=None):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.plane_image_path = image_path
        self.cameraMatrix = cameraMatrix
        self.plane_width = plane_width
        self.plane_height = plane_height

        if bkg_color is None:
            bkg_color = [154, 188, 255, 255]
        self.bkg_color = bkg_color

        if camera_rotation is None:
            camera_rotation = Rotation.from_rotvec([0, 0, 0], degrees=True)
        self.camera_rotation = camera_rotation

        if camera_translation is None:
            camera_translation = np.zeros((3,))
        self.camera_translation = camera_translation

        self.f = np.array([cameraMatrix[0, 0], cameraMatrix[1, 1]])
        self.c = cameraMatrix[:2, 2]

        self.init_vtk()

    def generate_image_with_obj_at_transform(self, plane_translation: np.array, plane_rotation: Rotation, save_path: str):
        self.plane = vtkPlaneSource()
        self.plane.SetOrigin(-self.plane_width * 0.5, -self.plane_height * 0.5, 0.0)
        self.plane.SetPoint1(self.plane_width * 0.5, -self.plane_height * 0.5, 0.0)
        self.plane.SetPoint2(-self.plane_width * 0.5, self.plane_height * 0.5, 0.0)
        self.plane.SetCenter(plane_translation[0], plane_translation[1], plane_translation[2])
        rotVec = plane_rotation.as_rotvec(degrees=True)
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
        writer.SetFileName(save_path)
        writer.SetInputData(w2if.GetOutput())
        writer.Write()
        self.renderer.RemoveActor(planeActor)

    def init_vtk(self):
        self.colors = vtkNamedColors()
        self.colors.SetColor('BkgColor', self.bkg_color)

        readerFactory = vtkImageReader2Factory()
        textureFile = readerFactory.CreateImageReader2(self.plane_image_path)
        textureFile.SetFileName(self.plane_image_path)
        textureFile.Update()

        self.textureMap = vtkTexture()
        self.textureMap.SetInputConnection(textureFile.GetOutputPort())
        self.textureMap.InterpolateOff()

        self.renderer = vtkRenderer()
        self.renWin = vtkRenderWindow()
        self.renWin.AddRenderer(self.renderer)
        self.renWin.SetShowWindow(False)

        self.renderer.SetBackground(self.colors.GetColor3d('BkgColor'))
        self.renWin.SetSize(self.image_width, self.image_height)

        self.renderer.ResetCamera()
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(self.camera_translation[0], self.camera_translation[1], self.camera_translation[2])
        focalPoint = self.camera_rotation.apply([0, 0, 1])
        cam.SetFocalPoint(self.camera_translation[0] + focalPoint[0], self.camera_translation[1] + focalPoint[1],
                          self.camera_translation[2] + focalPoint[2])
        viewUp = self.camera_rotation.apply([0, -1, 0])
        cam.SetViewUp(self.camera_translation[0] + viewUp[0], self.camera_translation[1] + viewUp[1],
                      self.camera_translation[2] + viewUp[2])
        wcx = -2.0 * (self.c[0] - self.image_width / 2.0) / self.image_width
        wcy = 2.0 * (self.c[1] - self.image_width / 2.0) / self.image_height
        cam.SetWindowCenter(wcx, wcy)
        angle = 180 / np.pi * 2.0 * np.arctan2(self.image_height / 2.0, self.f[1])
        cam.SetViewAngle(angle)
        m = np.eye(4)
        aspect = self.f[1] / self.f[0]
        m[0, 0] = 1.0 / aspect
        t = vtk.vtkTransform()
        t.SetMatrix(m.flatten())
        cam.SetUserTransform(t)
        self.renderer.ResetCameraClippingRange()
