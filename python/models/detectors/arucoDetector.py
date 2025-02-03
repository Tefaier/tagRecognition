import numpy as np
import cv2
import cv2.aruco as aruco

from python.models.detectors.detector import TagDetector

class ArucoDetector(TagDetector):
    def __init__(self, cameraMatrix: np.ndarray, distortionCoefficients: np.ndarray, tagLength: float, settings: cv2.aruco.DetectorParameters, tagFamily: int, name: str = 'aruco'):
        super().__init__(name, cameraMatrix, distortionCoefficients)
        self.dictionary = aruco.getPredefinedDictionary(tagFamily)
        self.settings = settings
        self.detector = aruco.ArucoDetector(self.dictionary, self.settings)
        self.objPoints = np.array([[-tagLength / 2, tagLength / 2, 0],
                              [tagLength / 2, tagLength / 2, 0],
                              [tagLength / 2, -tagLength / 2, 0],
                              [-tagLength / 2, -tagLength / 2, 0]], dtype=np.float32)
        self.tagLength = tagLength

    def detectObjectPoints(self, image: np.ndarray) -> (list, list):
        markerCorners, _, _ = self.detector.detectMarkers(image)
        return self.objPoints.copy(), markerCorners[0].reshape((4, 1, 2)) if len(markerCorners) > 0 else None

    def detect(self, image: np.ndarray) -> (list, list, list):
        markerCorners, markerIds, rejectedCandidates = self.detector.detectMarkers(image)

        ids = []
        rotations = []
        translations = []
        if markerIds is not None:
            for i in range(len(markerCorners)):
                success, rvec, tvec = cv2.solvePnP(self.objPoints, markerCorners[i], self.cameraMatrix, self.distortionCoefficients)
                rvec = rvec.reshape((3,))
                tvec = tvec.reshape((3,))
                if success:
                    ids.append(markerIds[i][0])
                    rotations.append(rvec)
                    translations.append(tvec)
        return translations, rotations, ids

    def detectorSettings(self) -> dict:
        return {

        }
