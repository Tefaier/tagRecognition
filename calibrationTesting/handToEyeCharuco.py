from pathlib import Path
import cv2 as cv2
import numpy.linalg
import vtk
import numpy as np
from scipy.spatial.transform import Rotation
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkPlaneSource
from vtkmodules.vtkIOImage import vtkImageReader2Factory, vtkPNGWriter
from vtkmodules.vtkRenderingCore import vtkTexture, vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow, \
    vtkWindowToImageFilter
from python.constantsForCheck import imageWidth, imageHeight, camMatrix, distortionCoefficients

cameraRotation = Rotation.from_rotvec([0, 0, 0], degrees=True)
cameraTranslation = np.array([0, 0, 0])
makeImages = True
displayDetectedAxis = True

print(f"Real rotation    {cameraRotation.as_rotvec(degrees=True)}")
print(f"Real translation {cameraTranslation.reshape((3,))}")

image = 'testImages/chessboard3.png'
resultSaveFolder = 'createdImages'
patternWidth = 0.1
patternHeight = 0.1 * 9 / 11
squareSize = patternWidth / 11
chessboardPattern = (8, 6)
axis = np.float32([[0,0,0], [0.05,0,0], [0,0.05,0], [0,0,0.05]]).reshape(-1,3)
objp = np.zeros((chessboardPattern[0] * chessboardPattern[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardPattern[0], 0:chessboardPattern[1]].T.reshape(-1, 2) * squareSize
objp[:, 0] -= squareSize * (chessboardPattern[0] - 1) / 2.0
objp[:, 1] -= squareSize * (chessboardPattern[1] - 1) / 2.0

class PlaneRenderer():
    def __init__(self, windowWidth, windowHeight, cameraMatrix, imagePath, bkgColor=None):
        if not makeImages: return
        if bkgColor is None:
            bkgColor = [154, 188, 255, 255]
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.cameraMatrix = cameraMatrix
        self.imagePath = imagePath
        self.bkgColor = bkgColor

        self.f = np.array([cameraMatrix[0, 0], cameraMatrix[1, 1]])
        self.c = cameraMatrix[:2, 2]

        self.init_vtk()

    def renderPlane(self, planeTranslation: np.array, planeRotation: Rotation, planeWidth: float, planeHeight: float, saveTo: str):
        if not makeImages: return
        self.plane = vtkPlaneSource()
        self.plane.SetOrigin(-planeWidth*0.5, -planeHeight*0.5, 0.0)
        self.plane.SetPoint1(planeWidth*0.5, -planeHeight*0.5, 0.0)
        self.plane.SetPoint2(-planeWidth*0.5, planeHeight*0.5, 0.0)
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
        cam.SetPosition(cameraTranslation[0], cameraTranslation[1], cameraTranslation[2])
        focalPoint = cameraRotation.apply([0, 0, 1])
        cam.SetFocalPoint(cameraTranslation[0] + focalPoint[0], cameraTranslation[1] + focalPoint[1], cameraTranslation[2] + focalPoint[2])
        viewUp = cameraRotation.apply([0, -1, 0])
        cam.SetViewUp(cameraTranslation[0] + viewUp[0], cameraTranslation[1] + viewUp[1], cameraTranslation[2] + viewUp[2])
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


def getResPath(index: int):
    return resultSaveFolder + '/' + str(index) + '.png'

def draw(img, imgpts):
    imgpts = imgpts.astype("int32")
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (0, 0, 255), 5)
    return img

def ensureFolderExists(relativePath: str):
    Path(relativePath).mkdir(parents=True, exist_ok=True)

ensureFolderExists(resultSaveFolder)
renderer = PlaneRenderer(imageWidth, imageHeight, camMatrix, image)

index = 0
translationsGlobal = []
rotationsGlobal = []

translation = np.array([0, 0, 1])
rotation = Rotation.from_rotvec([180, 0, 0], degrees=True)
translationsGlobal.append(translation)
rotationsGlobal.append(rotation.as_rotvec(degrees=False))
renderer.renderPlane(translation, rotation, patternWidth, patternHeight, getResPath(index))
index += 1

for angle in np.linspace(0, 360, 10):
    translation = np.array([0, 0, 1]) + Rotation.from_rotvec([0, 0, angle], degrees=True).apply([0, 0.4, 0])
    rotation = Rotation.from_rotvec([180, 0, 0], degrees=True)
    translationsGlobal.append(translation)
    rotationsGlobal.append(rotation.as_rotvec(degrees=False))
    renderer.renderPlane(translation, rotation, patternWidth, patternHeight, getResPath(index))
    index += 1

for angle in np.linspace(0, 180, 10):
    translation = np.array([0, 0, 1]) + Rotation.from_rotvec([0, angle, 0], degrees=True).apply([0.4, 0, 0])
    rotation = Rotation.from_rotvec([180, 0, 0], degrees=True) * Rotation.from_rotvec([0, 90 - angle, 0], degrees=True)
    translationsGlobal.append(translation)
    rotationsGlobal.append(rotation.as_rotvec(degrees=False))
    renderer.renderPlane(translation, rotation, patternWidth, patternHeight, getResPath(index))
    index += 1

for angle in np.linspace(0, 180, 10):
    translation = np.array([0, 0, 1]) + Rotation.from_rotvec([-angle, 0, 0], degrees=True).apply([0, 0.4, 0])
    rotation = Rotation.from_rotvec([180, 0, 0], degrees=True) * Rotation.from_rotvec([90 - angle, 0, 0], degrees=True)
    translationsGlobal.append(translation)
    rotationsGlobal.append(rotation.as_rotvec(degrees=False))
    renderer.renderPlane(translation, rotation, patternWidth, patternHeight, getResPath(index))
    index += 1

detectedMask = [True] * index
translationsCamera = []
rotationsCamera = []

for i in range(0, index):
    img = cv2.imread(getResPath(i))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    success, corners = cv2.findChessboardCorners(gray, chessboardPattern, None)
    if success == False:
        detectedMask[i] = False
        continue
    success, rvec, tvec = cv2.solvePnP(objp, corners, camMatrix, distortionCoefficients)
    translationsCamera.append(tvec.reshape((3,)))
    rotationsCamera.append(rvec.reshape((3,)))

    if displayDetectedAxis:
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camMatrix, distortionCoefficients)
        img = draw(img, imgpts)
        cv2.imshow('img', img)
        k = cv2.waitKey(0) & 0xFF



translationsGlobal = np.array(translationsGlobal)[detectedMask]
rotationsGlobal = np.array(rotationsGlobal)[detectedMask]
reverseRotations = [Rotation.from_rotvec(rot, degrees=False).inv() for rot in rotationsGlobal]
reverseTranslations = [-1 * reverseRotations[index].apply(tr) for index, tr in enumerate(translationsGlobal)]
reverseRotations = [rot.as_rotvec(degrees=False) for rot in reverseRotations]
rCamera, tCamera = cv2.calibrateHandEye(reverseRotations, reverseTranslations, rotationsCamera, translationsCamera, method=cv2.CALIB_HAND_EYE_PARK)
print(f"Obtained rotation    {Rotation.from_matrix(rCamera).as_rotvec(degrees=True)}")
print(f"Obtained translation {tCamera.reshape((3,))}")
rCamera, tCamera = cv2.calibrateHandEye(reverseRotations, reverseTranslations, rotationsCamera, translationsCamera, method=cv2.CALIB_HAND_EYE_TSAI)
print(f"Obtained rotation    {Rotation.from_matrix(rCamera).as_rotvec(degrees=True)}")
print(f"Obtained translation {tCamera.reshape((3,))}")
rCamera, tCamera = cv2.calibrateHandEye(reverseRotations, reverseTranslations, rotationsCamera, translationsCamera, method=cv2.CALIB_HAND_EYE_HORAUD)
print(f"Obtained rotation    {Rotation.from_matrix(rCamera).as_rotvec(degrees=True)}")
print(f"Obtained translation {tCamera.reshape((3,))}")
rCamera, tCamera = cv2.calibrateHandEye(reverseRotations, reverseTranslations, rotationsCamera, translationsCamera, method=cv2.CALIB_HAND_EYE_DANIILIDIS)
print(f"Obtained rotation    {Rotation.from_matrix(rCamera).as_rotvec(degrees=True)}")
print(f"Obtained translation {tCamera.reshape((3,))}")
rCamera, tCamera = cv2.calibrateHandEye(reverseRotations, reverseTranslations, rotationsCamera, translationsCamera, method=cv2.CALIB_HAND_EYE_ANDREFF)
print(f"Obtained rotation    {Rotation.from_matrix(rCamera).as_rotvec(degrees=True)}")
print(f"Obtained translation {tCamera.reshape((3,))}")