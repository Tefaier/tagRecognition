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
from python.utils import from_local_to_global


class VTKGenerator(ImageGenerator):
    camera_rotation: Rotation
    camera_translation: np.array

    def __init__(self, image_width, image_height, local_translations: list[np.array], local_rotations: list[Rotation], image_paths: list[str], cameraMatrix, plane_width: float, plane_height: float, bkg_color=None, camera_rotation: Rotation=None, camera_translation: np.array=None):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.local_translations = local_translations
        self.local_rotations = local_rotations
        self.image_paths = image_paths
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

    def generate_image_with_obj_at_transform(self, obj_translation: np.array, obj_rotation: Rotation, save_path: str) -> bool:
        planeActors = []
        for index in range(len(self.image_paths)):
            plane = vtkPlaneSource()
            plane.SetOrigin(-self.plane_width * 0.5, -self.plane_height * 0.5, 0.0)
            plane.SetPoint1(self.plane_width * 0.5, -self.plane_height * 0.5, 0.0)
            plane.SetPoint2(-self.plane_width * 0.5, self.plane_height * 0.5, 0.0)
            translation, rotation = from_local_to_global(obj_translation, obj_rotation, self.local_translations[index], self.local_rotations[index])
            plane.SetCenter(translation[0], translation[1], translation[2])
            rotVec = rotation.as_rotvec(degrees=True)
            plane.Rotate(numpy.linalg.norm(rotVec), (rotVec[0], rotVec[1], rotVec[2]))

            planeMapper = vtkPolyDataMapper()
            planeMapper.SetInputConnection(plane.GetOutputPort())

            planeActor = vtkActor()
            planeActor.SetMapper(planeMapper)
            planeActor.SetTexture(self.textureMaps[index])
            planeActor.GetProperty().LightingOff()
            self.renderer.AddActor(planeActor)
            planeActors.append(planeActor)

        w2if = vtkWindowToImageFilter()
        w2if.SetInput(self.renWin)
        w2if.Update()

        writer = vtkPNGWriter()
        writer.SetFileName(save_path)
        writer.SetInputData(w2if.GetOutput())
        writer.Write()
        for actor in planeActors:
            self.renderer.RemoveActor(actor)
        return True

    def generate_images_with_obj_at_transform(self, obj_translation: np.array, obj_rotation: Rotation, save_paths: list[str]) -> bool:
        for path in save_paths:
            self.generate_image_with_obj_at_transform(obj_translation, obj_rotation, path)
        return True

    def check_transform_is_available(self, obj_translation: np.array, obj_rotation: Rotation) -> bool:
        return True

    def init_vtk(self):
        self.colors = vtkNamedColors()
        self.colors.SetColor('BkgColor', self.bkg_color)

        readerFactory = vtkImageReader2Factory()
        self.textureMaps = []
        for index in range(len(self.image_paths)):
            textureFile = readerFactory.CreateImageReader2(self.image_paths[index])
            textureFile.SetFileName(self.image_paths[index])
            textureFile.Update()
            self.textureMaps.append(vtkTexture())
            self.textureMaps[-1].SetInputConnection(textureFile.GetOutputPort())
            self.textureMaps[-1].InterpolateOff()

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
