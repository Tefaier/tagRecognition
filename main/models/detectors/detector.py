import numpy as np

# considers default tag orientation on image to be
# x-to right
# y-to up
# z-to viewer
class TagDetector:
    name: str

    def __init__(self, name: str, cameraMatrix: np.ndarray, distortionCoefficients: np.ndarray):
        self.name = name
        self.cameraMatrix = cameraMatrix
        self.distortionCoefficients = distortionCoefficients

    # returns (objpoints, imagepoints)
    def detectObjectPoints(self, image: np.ndarray, tagLength: float) -> (list, list):
        pass

    # returns (translations, rotations, ids)
    def detect(self, image: np.ndarray, tagLength: float) -> (list, list, list):
        pass
