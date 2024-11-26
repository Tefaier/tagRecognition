from math import degrees

import numpy as np
import cv2
import cv2.aruco as aruco
from dt_apriltags import Detector, Detection
from scipy.spatial.transform import Rotation

from checkAlgo.constantsForCheck import camMatrix, distortionCoefficients


def getGrayImage(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)
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
                rvec = rvec.reshape((3,))
                tvec = tvec.reshape((3,))
                if success:
                    ids.append(markerIds[i])
                    rotation = Rotation.from_rotvec(rvec, degrees=False)
                    rotationVector = Rotation.from_rotvec(rotation.apply([0.0, 180.0, 0.0]), degrees=True)
                    rotation = rotationVector * rotation
                    rotations.append(rotation.as_rotvec(degrees=False))
                    transforms.append(tvec)
        return (transforms, rotations, ids)


class AlgoApriltag(Algo):
    # fx - x focal length in pixels
    # fy - y focal length in pixels
    # cx - x of focal center in pixels
    # cy - y of focal center in pixels
    def __init__(self, name: str, camMatrix: np.ndarray, distCoeffs: np.ndarray, tagFamily: str = "tag36h11"):
        super().__init__(name)
        self.fx = camMatrix[0, 0]
        self.fy = camMatrix[1, 1]
        self.cx = camMatrix[0, 2]
        self.cy = camMatrix[1, 2]
        self.camMatrix = camMatrix
        self.distCoeffs = distCoeffs
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
        imageGray = cv2.undistort(imageGray, self.camMatrix, self.distCoeffs, None)
        results = self.detector.detect(
            imageGray,
            estimate_tag_pose=True,
            camera_params=[self.fx, self.fy, self.cx, self.cy],
            tag_size=markerLength)
        ids = []
        rotations = []
        transforms = []
        for r in results:
            ids.append(r.tag_id)
            rotation = Rotation.from_matrix(r.pose_R)
            rotations.append(rotation.as_rotvec(degrees=False))
            transforms.append([val[0] for val in r.pose_t])
        return (transforms, rotations, ids)

# for ARUCO rotation is rotation vector
# also for it z is out of the tag, x is to the right, y is to the top
# it suggests camera z is forward, x is to the right, y is down
arucoDetector = AlgoAruco(
    name="aruco",
    camMatrix=camMatrix,
    distCoeffs=distortionCoefficients
)

apriltagDetector = AlgoApriltag(
    name="apriltag",
    camMatrix=camMatrix,
    distCoeffs=distortionCoefficients
)