import numpy as np
from scipy.spatial.transform import Rotation

# considers default camera orientation in image frame to be
# x-to right
# y-to down
# z-into image
class ImageGenerator:
    def __init__(self):
        pass

    def makeImageWithPlane(self, planeTranslation: np.array, planeRotation: Rotation, savePath: str):
        pass
