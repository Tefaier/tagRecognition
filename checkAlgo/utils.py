import ast

import numpy as np
from pandas import Series
from scipy.spatial.transform import Rotation

def parseRotation(rotation: list) -> Rotation:
    if (len(rotation) == 0): return None
    rotation = np.array(rotation)
    if rotation.size == 9:
        return Rotation.from_matrix(rotation)
    else:
        return Rotation.from_rotvec(rotation)

def getRotationEuler(rotation: list, part: str, degrees: bool = False) -> float:
    rotation = parseRotation(rotation)
    parts = rotation.as_euler('xyz', degrees=degrees)
    return float(parts[0 if part == 'x' else (1 if part == 'y' else 2)])

def readStringOfList(listStr: Series) -> list:
    return [ast.literal_eval(lis.replace("np.float64(", '').replace(")", '')) for lis in listStr.values]