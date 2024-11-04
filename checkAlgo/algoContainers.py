import numpy as np
import cv2
import cv2.aruco as aruco
from dt_apriltags import Detector, Detection

def getGrayImage(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

class Algo:
    def detect(self, image: np.ndarray, markerLength: float) -> (list, list):
        print("Base class")

class AlgoAruco(Algo):
    def __init__(self, camMatrix: np.ndarray, distCoeffs: np.ndarray):
        super().__init__()
        self.camMatrix = camMatrix
        self.distCoeffs = distCoeffs
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.detectorParams = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.dictionary, self.detectorParams)

    def detect(self, image: np.ndarray, markerLength: float) -> (list, list):
        markerCorners, markerIds, rejectedCandidates = self.detector.detectMarkers(image)

        # координаты углов маркера в его собственной системе координат
        objPoints = np.array([[-markerLength / 2, markerLength / 2, 0],
                              [markerLength / 2, markerLength / 2, 0],
                              [markerLength / 2, -markerLength / 2, 0],
                              [-markerLength / 2, -markerLength / 2, 0]])

        rotations = []
        transforms = []
        if markerIds is not None:
            for i in range(len(markerCorners)):
                success, rvec, tvec = cv2.solvePnP(objPoints, markerCorners[i], self.camMatrix, self.distCoeffs)
                if success:
                    rotations.append(rvec)
                    transforms.append(tvec)
        return (rotations, transforms)


class AlgoApriltag(Algo):
    # fx - x focal length in pixels
    # fy - y focal length in pixels
    # cx - x of focal center in pixels
    # cy - y of focal center in pixels
    def __init__(self, fx: float, fy: float, cx: float, cy: float, tagFamily: str = "tagStandard41h12"):
        super().__init__()
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

    def detect(self, image: np.ndarray, markerLength: float) -> (list, list):
        imageGray = getGrayImage(image)
        results = self.detector.detect(
            imageGray,
            estimate_tag_pose=True,
            camera_params=[self.fx, self.fy, self.cx, self.cy],
            tag_size=markerLength)
        rotations = []
        transforms = []
        for r in results:
            rotations.append(r.pose_R)
            transforms.append(r.pose_t)
        return (rotations, transforms)

arucoDetector = AlgoAruco(
    camMatrix=np.array(
        [[804.7329058535828, 0.0, 549.3237487667773],
         [0.0, 802.189566021595, 293.62680986426403],
         [0.0, 0.0, 1.0]]),
    distCoeffs=np.array(
        [-0.12367717208987415, 1.3006314330799533, -0.00045665885332229637, -0.028794247586331707, -2.264152794148503]))

apriltagDetector = AlgoApriltag(fx=1, fy=1, cx=1, cy=1)