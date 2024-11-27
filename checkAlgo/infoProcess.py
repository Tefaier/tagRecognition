import pandas as pd
import numpy as np
from numpy.linalg import norm
from checkAlgo.constantsForCheck import resultFolder, analiseFile, detectionFile, acceptedTransformError, \
    acceptedRotationError
from checkAlgo.utils import parseRotation, readStringOfList


def getVectorError(vector1: list, vector2: list) -> float:
    if (len(vector1) == 0 or len(vector2) == 0): return -1.0
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    norm1 = norm(vector1)
    norm2 = norm(vector2 - vector1)
    return float(norm2)  # / norm1 if norm1 != 0 else 1

def getRotationError(rotation1: list, rotation2: list) -> float:
    rotation1 = parseRotation(rotation1)
    rotation2 = parseRotation(rotation2)
    if (rotation1 == None or rotation2 == None): return -1.0

    rotation1To2 = rotation2 * rotation1.inv()
    return float(norm(rotation1To2.as_rotvec(degrees=False)))

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
    transformPass = errorT[-1] <= acceptedTransformError
    rotationPass = errorR[-1] <= acceptedRotationError
    isSuccess[i] = transformPass and rotationPass

toAnalise['isSuccess'] = isSuccess
toAnalise['errorT'] = errorT
toAnalise['errorR'] = errorR
toAnalise.drop(columns=['arucoAvailable'])
toAnalise.to_csv(resultFolder + "/" + analiseFile, index=False)