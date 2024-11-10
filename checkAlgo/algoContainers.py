import numpy as np
import cv2
import cv2.aruco as aruco
from dt_apriltags import Detector, Detection

from checkAlgo.constantsForCheck import camMatrix


def getGrayImage(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

class Algo:
    name: str

    def __init__(self, name: str):
        self.name = name

    def detect(self, image: np.ndarray, markerLength: float) -> (list, list, list):
        print("Base class")

class AlgoAruco(Algo):
    def __init__(self, name: str, camMatrix: np.ndarray):
        super().__init__(name)
        self.camMatrix = camMatrix
        self.distCoeffs = np.array([0, 0, 0, 0, 0])
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
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


class AlgoApriltag(Algo):
    # fx - x focal length in pixels
    # fy - y focal length in pixels
    # cx - x of focal center in pixels
    # cy - y of focal center in pixels
    def __init__(self, name: str, fx: float, fy: float, cx: float, cy: float, tagFamily: str = "tagStandard41h12 tag25h9"):
        super().__init__(name)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.detector = Detector(
            searchpath=['apriltags'],
            families=tagFamily,  # tagStandard41h12 tag25h9 tag36h11
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0)

    def detect(self, image: np.ndarray, markerLength: float) -> (list, list, list):
        imageGray = getGrayImage(image)
        results = self.detector.detect(
            imageGray,
            estimate_tag_pose=True,
            camera_params=[self.fx, self.fy, self.cx, self.cy],
            tag_size=markerLength)
        ids = []
        rotations = []
        transforms = []
        for r in results:
            ids.append(r.id)
            rotations.append(r.pose_R)
            transforms.append(r.pose_t)
        return (rotations, transforms, ids)

arucoDetector = AlgoAruco(
    name="aruco",
    camMatrix=camMatrix)

apriltagDetector = AlgoApriltag(
    name="apriltag",
    fx=camMatrix[0, 0],
    fy=camMatrix[1, 1],
    cx=camMatrix[0, 2],
    cy=camMatrix[1, 2])