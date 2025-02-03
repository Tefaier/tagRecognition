from typing import Sequence

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from python.models.detectors.detector import TagDetector
from python.utils import getGrayImage

# expects chessboard pattern to be centered at the image with it
# considers chessboard orientation on image to be (rectify applied)
# x-to right
# y-to down
# z-from viewer
class ChessboardDetector(TagDetector):
    def __init__(self, cameraMatrix: np.ndarray, distortionCoefficients: np.ndarray, pattern: Sequence[int], squareSize: float, name: str = 'chessboard'):
        super().__init__(name, cameraMatrix, distortionCoefficients)
        self.chessboardPattern = pattern
        self.squareSize = squareSize
        self.objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * squareSize
        self.objp[:, 0] -= squareSize * (pattern[0] - 1) / 2.0
        self.objp[:, 1] -= squareSize * (pattern[1] - 1) / 2.0
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def rectifyRotation(self, rotation: Rotation):
        extraRotation = Rotation.from_rotvec(rotation.apply([180, 0, 0]), degrees=True)
        return rotation * extraRotation

    def detectObjectPoints(self, image: np.ndarray) -> (list, list):
        gray = getGrayImage(image)
        success, corners = cv2.findChessboardCorners(gray, self.chessboardPattern, None)
        if not success: return self.objp, None
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        return self.objp, corners

    def detect(self, image: np.ndarray) -> (list, list, list):
        gray = getGrayImage(image)
        success, corners = cv2.findChessboardCorners(gray, self.chessboardPattern, None)
        if not success: return [], [], None
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        success, rvec, tvec = cv2.solvePnP(self.objp, corners, self.cameraMatrix, self.distortionCoefficients)
        rvec = rvec.reshape((3,))
        rvec = self.rectifyRotation(Rotation.from_rotvec(rvec, degrees=False)).as_rotvec(degrees=False)
        return tvec.reshape((1,3)), rvec.reshape((1,3)), None

    def detectorSettings(self) -> dict:
        return {}