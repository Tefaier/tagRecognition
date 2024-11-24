import os
from math import degrees

import cv2
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation

from checkAlgo.constantsForCheck import collectionFolder, csvName, tagLength, tagImagesFolder

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

def getImageWithParams(transform: list, rotation: Rotation, tagSize: float, tagImage: np.ndarray) -> np.ndarray:
    pass

def makeOutput(index: int, img: np.ndarray, transform: list, rotation: list, isAruco: bool = False, extraInfo: dict = None):
    if extraInfo is None:
        extraInfo = {}
    cv2.imwrite(filename=(collectionFolder + "/" + str(toWriteFrom + index) + ".png"), img=img)
    # fill values
    imageNames.append(toWriteFrom + index)
    arucoAvailables.append(isAruco)
    transforms.append(transform)
    rotations.append(rotation)
    otherInfos.append(extraInfo)

tagImage = cv2.imread(tagImagesFolder + '/' + 'aruco_1.svg')

for i in range(0, 0):
    transform = [0, 0, 10]
    rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    image = getImageWithParams(transform, rotation, tagLength, tagImage)
    makeOutput(iterationIndex, image, transform, rotation.as_matrix().tolist(), True, extraInfo={'tagLength': tagLength, 'tagFamily': 'aruco5x5', 'tagId': 2})
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