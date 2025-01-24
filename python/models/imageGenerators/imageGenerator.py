import numpy as np
from scipy.spatial.transform import Rotation

# considers default camera orientation in image frame to be
# x-to right
# y-to down
# z-into image
class ImageGenerator:
    def __init__(self, imageWidth, imageHeight, planeImagePath):
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.planeImagePath = planeImagePath

    def makeImageWithPlane(self, planeTranslation: np.array, planeRotation: Rotation, imagePath: str):
        pass
