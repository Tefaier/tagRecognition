import cv2 as cv2
from cv2 import aruco
import numpy as np
from scipy.spatial.transform import Rotation

from checkAlgo.constantsForCheck import imageWidth, imageHeight, camMatrix, distortionCoefficients
from checkAlgo.virtualCamera import PlaneRenderer

image = 'testImages/aruco_5x5_2.png'
resultSaveFolder = 'createdImages'
chessboardSize = 0.1
objp = np.zeros((7*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2) * (chessboardSize / 8)
objp[:, 0:2] -= (chessboardSize * 3 / 8)
n, k = 30, 0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, n, k)
axis = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)

class Algo:
    name: str

    def __init__(self, name: str):
        self.name = name

    def detect(self, image: np.ndarray, markerLength: float) -> (list, list, list):
        print("Base class")

class AlgoAruco(Algo):
    def __init__(self, name: str, camMatrix: np.ndarray, distCoeffs: np.ndarray, tagFamily: int):
        super().__init__(name)
        self.camMatrix = camMatrix
        self.distCoeffs = distCoeffs
        self.dictionary = aruco.getPredefinedDictionary(tagFamily)
        self.detectorParams = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.dictionary, self.detectorParams)

    def detect(self, image: np.ndarray, markerLength: float) -> (list, list, list):
        markerCorners, markerIds, rejectedCandidates = self.detector.detectMarkers(image)

        # координаты углов маркера в его собственной системе координат
        objPoints = np.array([[-markerLength / 2, markerLength / 2, 0],
                              [markerLength / 2, markerLength / 2, 0],
                              [markerLength / 2, -markerLength / 2, 0],
                              [-markerLength / 2, -markerLength / 2, 0]])

        ids = []
        rotations = []
        translations = []
        if markerIds is not None:
            for i in range(len(markerCorners)):
                success, rvec, tvec = cv2.solvePnP(objPoints, markerCorners[i], self.camMatrix, self.distCoeffs)
                rvec = rvec.reshape((3,))
                tvec = tvec.reshape((3,))
                if success:
                    ids.append(markerIds[i])
                    rotations.append(rvec)
                    translations.append(tvec)
        return (translations, rotations, ids)

def getResPath(index: int):
    return resultSaveFolder + '/' + str(index) + '.png'

def draw(img, imgpts):
    imgpts = imgpts.astype("int32")
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (0, 0, 255), 5)
    return img

renderer = PlaneRenderer(imageWidth, imageHeight, camMatrix, image)

index = 0
translationsGlobal = []
rotationsGlobal = []

translation = np.array([0, 0, 1])
rotation = Rotation.from_rotvec([180, 0, 0], degrees=True)
translationsGlobal.append(translation)
rotationsGlobal.append(rotation)
renderer.renderPlane(translation, rotation, chessboardSize, getResPath(index))
index += 1

for angle in np.linspace(0, 360, 10):
    translation = np.array([0, 0, 1]) + Rotation.from_rotvec([0, 0, angle], degrees=True).apply([0, 0.4, 0])
    rotation = Rotation.from_rotvec([180, 0, 0], degrees=True)
    translationsGlobal.append(translation)
    rotationsGlobal.append(rotation)
    renderer.renderPlane(translation, rotation, chessboardSize, getResPath(index))
    index += 1

for angle in np.linspace(0, 180, 10):
    translation = np.array([0, 0, 1]) + Rotation.from_rotvec([0, angle, 0], degrees=True).apply([0.4, 0, 0])
    rotation = Rotation.from_rotvec([180, 0, 0], degrees=True) * Rotation.from_rotvec([0, 90 - angle, 0], degrees=True)
    translationsGlobal.append(translation)
    rotationsGlobal.append(rotation)
    renderer.renderPlane(translation, rotation, chessboardSize, getResPath(index))
    index += 1

detector = AlgoAruco('aruco', camMatrix, distortionCoefficients, aruco.DICT_5X5_50)
detectedMask = [True] * index
translationsCamera = []
rotationsCamera = []

for i in range(0, index):
    img = cv2.imread(getResPath(i))
    tvec, rvec, ids = detector.detect(img, chessboardSize)
    if len(ids) == 0:
        detectedMask[i] = False
        continue
    tvec = tvec[0]
    rvec = rvec[0]
    translationsCamera.append(tvec)
    rotationsCamera.append(rvec)
'''
    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camMatrix, distortionCoefficients)
    img = draw(img, imgpts)
    cv2.imshow('img', img)
    k = cv2.waitKey(0) & 0xFF
'''

translationsGlobal = np.array(translationsGlobal)[detectedMask]
rotationsGlobal = np.array(rotationsGlobal)[detectedMask]
rCamera, tCamera = cv2.calibrateHandEye(rotationsGlobal, translationsGlobal, rotationsCamera, translationsCamera)
print(rCamera)
print(tCamera)
