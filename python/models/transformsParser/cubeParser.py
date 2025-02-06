import numpy as np
from scipy.spatial.transform import Rotation

from python.models.transformsParser.transformsParser import TransformsParser

# for a set of tags on top of a cube in an order of x+, y+, z+, x-, y-, z-
# image alignment of x axis is towards cubes y+, x-, y+, y-, x+, x+
class CubeParser(TransformsParser):
    def __init__(self, ids: list, cubeSize: float):
        x_rotate = Rotation.from_rotvec([90, 0, 0], degrees=True)
        y_rotate = Rotation.from_rotvec([0, 90, 0], degrees=True)
        z_rotate = Rotation.from_rotvec([0, 0, 90], degrees=True)
        x_vector = np.array([cubeSize, 0, 0])
        y_vector = np.array([0, cubeSize, 0])
        z_vector = np.array([0, 0, cubeSize])
        translations = [x_vector, y_vector, z_vector, -x_vector, -y_vector, -z_vector]
        rotations = [x_rotate * y_rotate, y_vector * y_rotate * x_rotate.inv(), z_rotate, z_rotate.inv() * x_rotate, x_rotate, x_rotate * x_rotate]
        super().__init__(translations, rotations, ids)
