# read csv
# for entry perform action
import cv2
import pandas as pd
import numpy as np

from checkAlgo.algoContainers import arucoDetector
from checkAlgo.constantsForCheck import csvName, outputFile, folderName

toCheck = pd.read_csv(folderName + "/" + csvName)

toCheck.rename(columns={"transform": "realT", "rotation": "realR"})
size = toCheck.shape[1]
toCheck = toCheck.reset_index()
method = np.full((size,), "aruko")
detectedT = []
detectedR = []
markerLength = 0.0525

# for now expect one detection per image
for index, row in toCheck.iterrows():
    t, r = arucoDetector.detect(image=cv2.imread(folderName + "/" + row["imageName"]), markerLength=markerLength)
    detectedT.append(t[0] if len(t) > 0 else [])
    detectedR.append(r[0] if len(r) > 0 else [])

toCheck["method"] = method
toCheck["detectedT"] = detectedT
toCheck["detectedR"] = detectedR
toCheck.to_csv(outputFile)