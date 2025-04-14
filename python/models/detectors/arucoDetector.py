import numpy as np
import cv2
import cv2.aruco as aruco

from python.models.detectors.detector import TagDetector

class ArucoDetector(TagDetector):
    def __init__(
            self,
            cameraMatrix: np.ndarray,
            distortionCoefficients: np.ndarray,
            tagSize: float,
            settings: cv2.aruco.DetectorParameters,
            tagFamily: int,
            name: str = 'aruco'
    ):
        super().__init__(name, cameraMatrix, distortionCoefficients)
        self.dictionary = aruco.getPredefinedDictionary(tagFamily)
        self.settings = settings
        self.detector = aruco.ArucoDetector(self.dictionary, self.settings)
        self.objPoints = np.array([[-tagSize / 2, tagSize / 2, 0],
                              [tagSize / 2, tagSize / 2, 0],
                              [tagSize / 2, -tagSize / 2, 0],
                              [-tagSize / 2, -tagSize / 2, 0]], dtype=np.float32)
        self.tagSize = tagSize

    def detect_object_points(self, image: np.ndarray) -> (list, list):
        marker_corners, _, _ = self.detector.detectMarkers(image)
        return self.objPoints.copy(), marker_corners[0].reshape((4, 1, 2)) if len(marker_corners) > 0 else None

    def detect(self, image: np.ndarray) -> (list, list, list):
        marker_corners, marker_ids, rejected_candidates = self.detector.detectMarkers(image)

        ids = []
        rotations = []
        translations = []
        if marker_ids is not None:
            for i in range(len(marker_corners)):
                success, rvec, tvec = cv2.solvePnP(self.objPoints, marker_corners[i], self.cameraMatrix, self.distortionCoefficients)
                rvec = rvec.reshape((3,))
                tvec = tvec.reshape((3,))
                if success:
                    ids.append(marker_ids[i][0])
                    rotations.append(rvec)
                    translations.append(tvec)
        return translations, rotations, ids

    def detector_settings(self) -> dict:
        # TODO decide on settings parameters that are used (changed for analyze)
        return {
            "aruco3": self.settings.useAruco3Detection,
            "cornerRefinementMinAccuracy ": self.settings.cornerRefinementMinAccuracy,
        }
