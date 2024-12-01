import pandas as pd
import numpy as np
from numpy.linalg import norm
from checkAlgo.constantsForCheck import resultFolder, analiseFile, detectionFile, acceptedTransformError, \
    acceptedRotationError
from checkAlgo.utils import parseRotation, readStringOfList


def getVectorError(vector1: list, vector2: list) -> list:
    if (len(vector1) == 0 or len(vector2) == 0): return []
    return [vector2[i] - vector1[i] for i in range(0, len(vector1))]


def getRotationError(rotation1: list, rotation2: list) -> list:
    rotation1 = parseRotation(rotation1)
    rotation2 = parseRotation(rotation2)
    if (rotation1 == None or rotation2 == None): return []

    rotation1To2 = rotation2 * rotation1.inv()
    return rotation1To2.as_rotvec(degrees=False).tolist()

# usecols=["imageName", "arucoAvailable", "method", "realT", "realR", "detectedT", "detectedR"]
toAnalise = pd.read_csv(resultFolder + "/" + detectionFile)

realT = readStringOfList(toAnalise['realT'])
realR = readStringOfList(toAnalise['realR'])
detectedT = readStringOfList(toAnalise['detectedT'])
detectedR = readStringOfList(toAnalise['detectedR'])
errorT = []
errorR = []
isSuccess = np.full((len(realT),), False)

for i in range(0, len(realT)):
    errorT.append(getVectorError(realT[i], detectedT[i]))
    errorR.append(getRotationError(realR[i], detectedR[i]))
    transformPass = len(errorT[-1]) != 0
    rotationPass = len(errorR[-1]) != 0
    isSuccess[i] = transformPass and rotationPass

toAnalise['isSuccess'] = isSuccess
toAnalise['errorT'] = errorT
toAnalise['errorR'] = errorR
toAnalise.drop(columns=['arucoAvailable'])
toAnalise.to_csv(resultFolder + "/" + analiseFile, index=False)