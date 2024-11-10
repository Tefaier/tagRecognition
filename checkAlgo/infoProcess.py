import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

from checkAlgo.constantsForCheck import resultFolder, analiseFile, detectionFile, acceptedTransformError, \
    acceptedRotationError

def getVectorError(vector1: list, vector2: list) -> float:
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    norm1 = norm(vector1)
    norm2 = norm(vector2 - vector1)
    return norm2 / norm1 if norm1 != 0 else 1

def getRotationError(rotation1: list, rotation2: list) -> float:
    rotation1 = Rotation.from_matrix(rotation1)
    rotation2 = Rotation.from_matrix(rotation2)
    rotation1To2 = rotation2 * rotation1.inv()
    return norm(rotation1To2.as_rotvec(degrees=True))

# usecols=["imageName", "arucoAvailable", "method", "realT", "realR", "detectedT", "detectedR"]
toAnalise = pd.read_csv(resultFolder + "/" + detectionFile)

realT = toAnalise['realT'].to_numpy()
realR = toAnalise['realR'].to_numpy()
detectedT = toAnalise['detectedT'].to_numpy()
detectedR = toAnalise['detectedR'].to_numpy()
isSuccess = np.full((realT.size,), False)

for i in range(0, realT.size):
    transformPass = getVectorError(realT[i], detectedT[i]) <= acceptedTransformError
    rotationPass = getRotationError(realR[i], realR[i]) <= acceptedRotationError
    isSuccess[i] = transformPass and rotationPass

toAnalise['isSuccess'] = isSuccess
toAnalise.drop(columns=['arucoAvailable', 'realT', 'realR', 'detectedT', 'detectedR'])
toAnalise.to_csv(resultFolder + "/" + analiseFile, index=False)