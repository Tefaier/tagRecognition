import os

import cv2
import pandas as pd
import numpy as np

from checkAlgo.constantsForCheck import collectionFolder, csvName

w = 100
h = 100
camMatrix = np.array([[804.7329058535828, 0.0, 549.3237487667773], [0.0, 802.189566021595, 293.62680986426403], [0.0, 0.0, 1.0]])
distCoeffs = np.array([-0.12367717208987415, 1.3006314330799533, -0.00045665885332229637, -0.028794247586331707, -2.264152794148503])
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camMatrix, distCoeffs, (w,h), 0, (w,h))

# fields: imageName, tagFamily, tagId, transform, rotation
imageNames = []
arucoAvailables = []
transforms = []
rotations = []
otherInfos = []

files = os.listdir(collectionFolder + "/")
files = list(filter(lambda name: name.split('.')[-1]=='png', files))
files = [int(name.split('.')[0]) for name in files]
toWriteFrom = max(files) + 1
iterationIndex = 0
for i in range(0, 0):
    image = cv2.imread("")
    image = cv2.undistort(image, camMatrix, distCoeffs, None, newcameramtx)
    # write image
    cv2.imwrite(filename=(collectionFolder + "/" + str(toWriteFrom + iterationIndex) + ".png"), img=image)
    # fill values
    imageNames.append(toWriteFrom + iterationIndex)
    arucoAvailables.append(True)
    transforms.append([0, 0, 0])
    rotations.append([0, 0, 0])
    otherInfos.append({})
    iterationIndex += 1

# creates DataFrame and appends it to file
collectedInfo = pd.DataFrame.from_dict({
    "imageName": imageNames,
    "arucoAvailable": arucoAvailables,
    "realT": transforms,
    "realR": rotations,
    "otherInfo": otherInfos
})
collectedInfo.to_csv(collectionFolder + "/" + csvName, header=False, mode='a', index=False)