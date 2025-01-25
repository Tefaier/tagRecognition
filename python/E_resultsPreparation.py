import os

import pandas as pd
import numpy as np
from numpy.linalg import norm
from python.constantsForCheck import resultFolder, analiseFile, detectionFile, acceptedTranslationError, \
    acceptedRotationError
from python.settings import generatedInfoFolder, detectionInfoFilename
from python.utils import parseRotation, readStringOfList


def getVectorError(vector1: list, vector2: list) -> list:
    if (len(vector1) == 0 or len(vector2) == 0): return []
    return [vector2[i] - vector1[i] for i in range(0, len(vector1))]


def getRotationError(rotation1: list, rotation2: list) -> list:
    rotation1 = parseRotation(rotation1)
    rotation2 = parseRotation(rotation2)
    if (rotation1 == None or rotation2 == None): return []

    rotation1To2 = rotation2 * rotation1.inv()
    return rotation1To2.as_rotvec(degrees=False).tolist()

def openAndPrepareRawInfo(path: str) -> pd.DataFrame:
    info = pd.read_csv(path)
    # info = info.reset_index()
    return info

def fillTransformErrors(dframe: pd.DataFrame):
    realT = readStringOfList(dframe['realT'])
    realR = readStringOfList(dframe['realR'])
    detectedT = readStringOfList(dframe['detectedT'])
    detectedR = readStringOfList(dframe['detectedR'])
    errorT = []
    errorR = []
    isSuccess = np.full((len(realT),), False)

    for i in range(0, len(realT)):
        errorT.append(getVectorError(realT[i], detectedT[i]))
        errorR.append(getRotationError(realR[i], detectedR[i]))
        translationPass = len(errorT[-1]) != 0
        rotationPass = len(errorR[-1]) != 0
        isSuccess[i] = translationPass and rotationPass

    dframe['isSuccess'] = isSuccess
    dframe['errorT'] = errorT
    dframe['errorR'] = errorR

def writeInfo(path: str, dframe: pd.DataFrame):
    if not os.path.exists(path):
        dframe.to_csv(path, header=True, mode='w', index=False)
        return
    df = pd.read_csv(path)
    pd.concat([df, dframe]).to_csv(path, header=True, mode='w', index=False)

def performPreparation(profile: str):
    profilePath = f"{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}"
    detectionInfo = openAndPrepareRawInfo(f"{profilePath}/{detectionInfoFilename}.csv")
    fillTransformErrors(detectionInfo)
    writeInfo(
        f"{profilePath}/{detectionInfoFilename}.csv",
        detectionInfo
    )

def testRun():
    performPreparation("test")

