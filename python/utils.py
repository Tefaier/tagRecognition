import ast
import json
import math
import os
import time
from pathlib import Path
from typing import Tuple, List, Any
import cv2

import numpy as np
import numpy.random
import scipy.stats
from pandas import Series
from scipy.spatial.transform import Rotation

from python.settings import generatedInfoFolder, generalInfoFilename

randomGenerator = numpy.random.Generator(np.random.default_rng(int(time.time())).bit_generator)

def axisToIndex(axis: str):
    return 0 if axis == 'x' else (1 if axis == 'y' else 2)

def parseRotation(rotation: list) -> Rotation:
    if (len(rotation) == 0): return None
    rotation = np.array(rotation)
    if rotation.size == 9:
        return Rotation.from_matrix(rotation)
    else:
        return Rotation.from_rotvec(rotation, degrees=False)

def getRotationEuler(rotation: list, part: str, degrees: bool = False) -> float:
    rotation = parseRotation(rotation)
    parts = rotation.as_euler('xyz', degrees=degrees)
    return float(parts[axisToIndex(part)])

def readStringOfList(listStr: Series) -> list:
    return [ast.literal_eval(lis.replace("np.float64(", '').replace(")", '')) for lis in listStr.values]

def readStringOfDict(listStr: Series) -> list[dict]:
    return [ast.literal_eval(lis.replace("np.float64(", '').replace(")", '')) for lis in listStr.values]

def generateNormalDistributionValue(center: float = 0, maxDeviation: float = 3) -> float:
    return min(
        max(
            -maxDeviation,
            randomGenerator.normal(loc=center, scale=maxDeviation / 3, size=None)),
        maxDeviation
    )

def deviateTransform(position: list, rotation: list, px: float = 0, py: float = 0, pz: float = 0, rx: float = 0, ry: float = 0, rz: float = 0) -> list[list[float]]:
    answer = [[], []]
    if (px == 0 and py == 0 and pz == 0):
        answer[0] = position
    else:
        answer[0] = [position[0] + px, position[1] + py, position[2] + pz]
    if (rx == 0 and ry == 0 and rz == 0):
        answer[1] = rotation
    else:
        answer[1] = [rotation[0] + rx, rotation[1] + ry, rotation[2] + rz]
    return answer

def ensureFolderExists(relativePath: str):
    Path(relativePath).mkdir(parents=True, exist_ok=True)

def getGrayImage(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def generateRandomNormVector() -> np.array:
    randomVector = (np.random.rand(3) * 2) - 1
    length = randomVector.dot(randomVector)
    if abs(length) < 1e-2:
        return generateRandomNormVector()
    return randomVector / math.sqrt(length)

def updateJSON(newInfo: dict, path: str):
    if os.path.exists(path):
        with open(path, 'r') as f:
            dic = json.load(f)
    else:
        dic = {}
    dic.update(newInfo)
    with open(path, 'w') as f:
        json.dump(dic, f)

def writeInfoToProfileJSON(profile: str, info: dict):
    path = f'{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}/{generalInfoFilename}.json'
    ensureFolderExists(f'{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}')
    updateJSON(info, path)

def readProfileJSON(profile: str) -> dict:
    path = f'{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}/{generalInfoFilename}.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            dic = json.load(f)
    else:
        dic = {}
    return dic

def copyCameraProfileInfo(fromProfile: str, toProfile: str):
    info = readProfileJSON(fromProfile)
    toLeaveList = ["cameraMatrix", "distortionCoefficients", "cameraTranslation", "cameraTranslation"]
    toLeaveDict = {}
    for key in toLeaveList:
        toLeaveDict[key] = info[key]
    writeInfoToProfileJSON(toProfile, toLeaveDict)
