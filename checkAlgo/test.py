import numpy as np
import cv2
import cv2.aruco as aruco
from checkAlgo.constantsForCheck import camMatrix, distortionCoefficients, tagLength


class Algo:
    name: str

    def __init__(self, name: str):
        self.name = name

    def detect(self, image: np.ndarray, markerLength: float) -> (list, list, list):
        print("Base class")

class AlgoAruco(Algo):
    def __init__(self, name: str, camMatrix: np.ndarray, distCoeffs: np.ndarray):
        super().__init__(name)
        self.camMatrix = camMatrix
        self.distCoeffs = distCoeffs
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
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
        transforms = []
        if markerIds is not None:
            for i in range(len(markerCorners)):
                success, rvec, tvec = cv2.solvePnP(objPoints, markerCorners[i], self.camMatrix, self.distCoeffs)
                if success:
                    ids.append(markerIds[i])
                    rotations.append(rvec)
                    transforms.append(tvec)
        return (rotations, transforms, ids)

arucoDetector = AlgoAruco(
    name="aruco",
    camMatrix=camMatrix,
    distCoeffs=distortionCoefficients
)

image = cv2.imread("collectedInfo/64.png")
result = arucoDetector.detect(image, tagLength)
print(result)