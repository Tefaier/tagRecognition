# read csv
# for entry perform action
import cv2
import pandas as pd
import numpy as np

from checkAlgo.algoContainers import arucoDetector
from checkAlgo.constantsForCheck import csvName, detectionFile, collectionFolder, resultFolder


# simple detection of one tag on image
def detectionToResult(transforms: list, rotations: list, ids: list) -> (list, list):
    if len(transforms) == 0:
        return ([], [])
    return (transforms[0], rotations[0])

toCheck = pd.read_csv(collectionFolder + "/" + csvName)

toCheck.rename(columns={"transform": "realT", "rotation": "realR"})
size = toCheck.shape[1]
toCheck = toCheck.reset_index()
method = np.full((size,), "aruko")
detectedT = []
detectedR = []
markerLength = 0.0525

# for now expect one detection per image
for index, row in toCheck.iterrows():
    t, r, ids = arucoDetector.detect(image=cv2.imread(collectionFolder + "/" + row["imageName"]), markerLength=markerLength)
    t, r = detectionToResult(t, r, ids)
    detectedT.append(t)
    detectedR.append(r)

toCheck["method"] = method
toCheck["detectedT"] = detectedT
toCheck["detectedR"] = detectedR
toCheck.to_csv(resultFolder + "/" + detectionFile)