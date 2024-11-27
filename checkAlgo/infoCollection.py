import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from checkAlgo.constantsForCheck import collectionFolder, csvName, tagLength, tagImagesFolder, imageWidth, imageHeight, camMatrix
from checkAlgo.virtualCamera import PlaneRenderer

# fields: imageName, tagFamily, tagId, transform, rotation
imageNames = []
arucoAvailables = []
transforms = []
rotations = []
otherInfos = []

files = os.listdir(collectionFolder + "/")
files = list(filter(lambda name: name.split('.')[-1]=='png', files))
files = [int(name.split('.')[0]) for name in files]
toWriteFrom = max(files, default=-1) + 1
iterationIndex = 0

tagImage = tagImagesFolder + '/' + '2.png'
renderer = PlaneRenderer(imageWidth, imageHeight, camMatrix, tagImage)

def getImageWithParams(transform: list, rotation: Rotation, tagSize: float, saveDestination: str):
    renderer.renderPlane(transform, rotation, tagSize, saveDestination)

def makeOutput(index: int, transform: list, rotation: list, isAruco: bool = False, extraInfo: dict = None):
    if extraInfo is None:
        extraInfo = {}
    # fill values
    imageNames.append(str(toWriteFrom + index) + str(".png"))
    arucoAvailables.append(isAruco)
    transforms.append([float(val) for val in transform])
    rotations.append(rotation)
    otherInfos.append(extraInfo)

def rotationWithRectify(toMake: Rotation) -> Rotation:
    rectify = Rotation.from_euler('xyz', [180, 0, 0], degrees=True)
    return toMake * rectify

for x in np.linspace(-80, 80, 100):
    transform = [0.0, 0.0, 0.1]
    rotation = Rotation.from_euler('xyz', [x, 0, 0], degrees=True)
    getImageWithParams(transform, rotationWithRectify(rotation), tagLength, collectionFolder + "/" + str(toWriteFrom + iterationIndex) + ".png")
    makeOutput(iterationIndex, transform, rotation.as_rotvec(degrees=False).tolist(), True, extraInfo={'tagLength': tagLength, 'tagFamily': 'tag36h11', 'tagId': 0})
    iterationIndex += 1

for y in np.linspace(-80, 80, 100):
    transform = [0.0, 0.0, 0.1]
    rotation = Rotation.from_euler('xyz', [0, y, 0], degrees=True)
    getImageWithParams(transform, rotationWithRectify(rotation), tagLength, collectionFolder + "/" + str(toWriteFrom + iterationIndex) + ".png")
    makeOutput(iterationIndex, transform, rotation.as_rotvec(degrees=False).tolist(), True, extraInfo={'tagLength': tagLength, 'tagFamily': 'tag36h11', 'tagId': 0})
    iterationIndex += 1

for posZ in np.linspace(0.1, 5, 100):
    transform = [0.0, 0.0, posZ]
    rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    getImageWithParams(transform, rotationWithRectify(rotation), tagLength, collectionFolder + "/" + str(toWriteFrom + iterationIndex) + ".png")
    makeOutput(iterationIndex, transform, rotation.as_rotvec(degrees=False).tolist(), True, extraInfo={'tagLength': tagLength, 'tagFamily': 'tag36h11', 'tagId': 0})
    iterationIndex += 1

for posY in np.linspace(-0.5, 0.5, 100):
    transform = [0.0, posY, 1]
    rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    getImageWithParams(transform, rotationWithRectify(rotation), tagLength, collectionFolder + "/" + str(toWriteFrom + iterationIndex) + ".png")
    makeOutput(iterationIndex, transform, rotation.as_rotvec(degrees=False).tolist(), True, extraInfo={'tagLength': tagLength, 'tagFamily': 'tag36h11', 'tagId': 0})
    iterationIndex += 1

# creates DataFrame and appends it to file
collectedInfo = pd.DataFrame.from_dict({
    "imageName": imageNames,
    "arucoAvailable": arucoAvailables,
    "realT": transforms,
    "realR": rotations,
    "otherInfo": otherInfos
})
collectedInfo.to_csv(collectionFolder + "/" + csvName, header=True, mode='w', index=False)