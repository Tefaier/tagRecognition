import numpy as np
from scipy.spatial.transform import Rotation

from python.models.transformsParser.transformsParser import TransformsParser

# for a set of tags on top of a cube in an order of x+, y+, z+, x-, y-, z-
# image alignment of x axis is towards cubes y+, x-, y+, y-, x+, x+
class CubeParser(TransformsParser):
    def __init__(self, ids: list, cubeSize: float):
        xRotate = Rotation.from_rotvec([90, 0, 0], degrees=True)
        yRotate = Rotation.from_rotvec([0, 90, 0], degrees=True)
        zRotate = Rotation.from_rotvec([0, 0, 90], degrees=True)
        xVector = np.array([cubeSize, 0, 0])
        yVector = np.array([0, cubeSize, 0])
        zVector = np.array([0, 0, cubeSize])
        translations = [xVector, yVector, zVector, -xVector, -yVector, -zVector]
        rotations = [xRotate * yRotate, yVector * yRotate * xRotate.inv(), zRotate, zRotate.inv() * xRotate, xRotate, xRotate * xRotate]
        super().__init__(translations, rotations, ids)
