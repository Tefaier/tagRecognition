# read csv
# for entry perform action
import cv2
import pandas as pd
import numpy as np

from checkAlgo.algoContainers import arucoDetector, apriltagDetector, Algo
from checkAlgo.constantsForCheck import csvName, detectionFile, collectionFolder, resultFolder, markerLength


# simple detection of one tag on image
def detectionToResult(transforms: list, rotations: list, ids: list) -> (list, list):
    if len(transforms) == 0:
        return ([], [])
    return (transforms[0], rotations[0])

def performDetection(method: Algo, dframe: pd.DataFrame, transformWrite: list, rotationWrite: list):
    for _, row in dframe.iterrows():
        t, r, ids = method.detect(image=cv2.imread(collectionFolder + "/" + row["imageName"]),
                                         markerLength=markerLength)
        t, r = detectionToResult(t, r, ids)
        transformWrite.append(t)
        rotationWrite.append(r)

def analyseInfo(method: Algo, dframe: pd.DataFrame, writeTo: str):
    methodData = np.full((dframe.shape[1],), method.name)
    detectedT = []
    detectedR = []

    performDetection(method, dframe, detectedT, detectedR)

    toCheck["method"] = methodData
    toCheck["detectedT"] = detectedT
    toCheck["detectedR"] = detectedR

    toCheck.drop(columns=['otherInfo'])  # not sure if to drop or not to drop
    toCheck.to_csv(writeTo, header=False, mode='a', index=False)

def openAndPrepareRawInfo(path: str) -> pd.DataFrame:
    info = pd.read_csv(path, usecols=["imageName", "arucoAvailable", "realT", "realR", "otherInfo"])
    # info = info.reset_index()
    return info

toCheck = openAndPrepareRawInfo(collectionFolder + "/" + csvName)
analyseInfo(arucoDetector, toCheck, resultFolder + "/" + detectionFile)
toCheck = openAndPrepareRawInfo(collectionFolder + "/" + csvName)
analyseInfo(apriltagDetector, toCheck, resultFolder + "/" + detectionFile)