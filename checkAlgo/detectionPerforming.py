# read csv
# for entry perform action
import cv2
import pandas as pd
import numpy as np

from checkAlgo.algoContainers import arucoDetector, apriltagDetector, Algo
from checkAlgo.constantsForCheck import csvName, detectionFile, collectionFolder, resultFolder, tagLength


methodDataFinal = []
detectedTFinal = []
detectedRFinal = []

# simple detection of one tag on image
def detectionToResult(transforms: list, rotations: list, ids: list) -> (list, list):
    if len(transforms) == 0:
        return ([], [])
    return (list(transforms[0]), list(rotations[0]))

def performDetection(method: Algo, dframe: pd.DataFrame, transformWrite: list, rotationWrite: list):
    for _, row in dframe.iterrows():
        t, r, ids = method.detect(image=cv2.imread(collectionFolder + "/" + row["imageName"]),
                                  markerLength=tagLength)
        t, r = detectionToResult(t, r, ids)
        transformWrite.append([float(val) for val in t])
        rotationWrite.append([float(val) for val in r])

def analyseInfo(method: Algo, dframe: pd.DataFrame):
    methodData = np.full((dframe.shape[0],), method.name)
    detectedT = []
    detectedR = []

    performDetection(method, dframe, detectedT, detectedR)

    methodDataFinal.extend(methodData)
    detectedTFinal.extend(detectedT)
    detectedRFinal.extend(detectedR)


def openAndPrepareRawInfo(path: str) -> pd.DataFrame:
    info = pd.read_csv(path, usecols=["imageName", "arucoAvailable", "realT", "realR", "otherInfo"])
    # info = info.reset_index()
    return info

def writeInfo(writeTo: str):
    if len(methodDataFinal) / toCheck.shape[0] == 2:
        toWrite = pd.concat([toCheck, toCheck], axis=0)
    else:
        toWrite = toCheck
    toWrite["method"] = methodDataFinal
    toWrite["detectedT"] = detectedTFinal
    toWrite["detectedR"] = detectedRFinal

    toWrite.drop(columns=['otherInfo'])  # not sure if to drop or not to drop
    toWrite.to_csv(writeTo, header=True, mode='w', index=False)

toCheck = openAndPrepareRawInfo(collectionFolder + "/" + csvName)
analyseInfo(arucoDetector, toCheck)
toCheck = openAndPrepareRawInfo(collectionFolder + "/" + csvName)
analyseInfo(apriltagDetector, toCheck)
writeInfo(resultFolder + "/" + detectionFile)