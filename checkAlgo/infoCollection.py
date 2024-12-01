import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from checkAlgo.constantsForCheck import collectionFolder, csvName, tagLength, tagImagesFolder, imageWidth, imageHeight, camMatrix
from checkAlgo.utils import deviateTransform, generateNormalDistributionValue
from checkAlgo.virtualCamera import PlaneRenderer

# fields: imageName, tagFamily, tagId, position, rotation
imageNames = []
arucoAvailables = []
positions = []
rotations = []
otherInfos = []

files = os.listdir(collectionFolder + "/")
files = list(filter(lambda name: name.split('.')[-1]=='png', files))
files = [int(name.split('.')[0]) for name in files]
toWriteFrom = max(files, default=-1) + 1
iterationIndex = 0

tagImage = tagImagesFolder + '/' + 'aruco_1.png'
ratioOfImageToTag = 9 / 7
renderer = PlaneRenderer(imageWidth, imageHeight, camMatrix, tagImage)

def getImageWithParams(position: list, rotation: Rotation, planeSize: float, saveDestination: str):
    renderer.renderPlane(position, rotation, planeSize, saveDestination)

def makeOutput(index: int, position: list, rotation: list, isAruco: bool = False, extraInfo: dict = None):
    if extraInfo is None:
        extraInfo = {}
    # fill values
    imageNames.append(str(toWriteFrom + index) + str(".png"))
    arucoAvailables.append(isAruco)
    positions.append([float(val) for val in position])
    rotations.append(rotation)
    otherInfos.append(extraInfo)

def rotationWithRectify(toMake: Rotation) -> Rotation:
    rectify = Rotation.from_euler('xyz', [180, 0, 0], degrees=True)
    return toMake * rectify

defaultPosition = [0.0, 0.0, 0.15]
samplesToGet = 50
samplesDispersion = 1

for x in np.linspace(-89, 89, 60):
    deviateValue = 89 * 2 / (60 * 2)
    rawPosition = defaultPosition
    rawRotation = [x, 0, 0]
    for i in range(0, samplesToGet):
        position, rotationEuler = deviateTransform(rawPosition, rawRotation, rx=generateNormalDistributionValue(maxDeviation=deviateValue))
        rotation = Rotation.from_euler('xyz', rotationEuler, degrees=True)
        getImageWithParams(position, rotationWithRectify(rotation), tagLength * ratioOfImageToTag, collectionFolder + "/" + str(toWriteFrom + iterationIndex) + ".png")
        makeOutput(iterationIndex, position, rotation.as_rotvec(degrees=False).tolist(), True, extraInfo={'tagLength': tagLength, 'tagFamily': 'tag36h11', 'tagId': 0})
        iterationIndex += 1

for y in np.linspace(-89, 89, 60):
    deviateValue = 89 * 2 / (60 * 2)
    rawPosition = defaultPosition
    rawRotation = [0, y, 0]
    for i in range(0, samplesToGet):
        position, rotationEuler = deviateTransform(rawPosition, rawRotation, ry=generateNormalDistributionValue(maxDeviation=deviateValue))
        rotation = Rotation.from_euler('xyz', rotationEuler, degrees=True)
        getImageWithParams(position, rotationWithRectify(rotation), tagLength * ratioOfImageToTag, collectionFolder + "/" + str(toWriteFrom + iterationIndex) + ".png")
        makeOutput(iterationIndex, position, rotation.as_rotvec(degrees=False).tolist(), True, extraInfo={'tagLength': tagLength, 'tagFamily': 'tag36h11', 'tagId': 0})
        iterationIndex += 1

for posZ in np.linspace(0.1, 4, 50):
    deviateValue = 4 / (50 * 2)
    rawPosition = [defaultPosition[0], defaultPosition[1], posZ]
    rawRotation = [0, 0, 0]
    for i in range(0, samplesToGet):
        position, rotationEuler = deviateTransform(rawPosition, rawRotation, pz=generateNormalDistributionValue(maxDeviation=deviateValue))
        rotation = Rotation.from_euler('xyz', rotationEuler, degrees=True)
        getImageWithParams(position, rotationWithRectify(rotation), tagLength * ratioOfImageToTag, collectionFolder + "/" + str(toWriteFrom + iterationIndex) + ".png")
        makeOutput(iterationIndex, position, rotation.as_rotvec(degrees=False).tolist(), True, extraInfo={'tagLength': tagLength, 'tagFamily': 'tag36h11', 'tagId': 0})
        iterationIndex += 1

for posY in np.linspace(-0.4, 0.4, 50):
    deviateValue = 0.8 / (50 * 2)
    rawPosition = [defaultPosition[0], posY, defaultPosition[2]]
    rawRotation = [0, 0, 0]
    for i in range(0, samplesToGet):
        position, rotationEuler = deviateTransform(rawPosition, rawRotation, py=generateNormalDistributionValue(maxDeviation=deviateValue))
        rotation = Rotation.from_euler('xyz', rotationEuler, degrees=True)
        getImageWithParams(position, rotationWithRectify(rotation), tagLength * ratioOfImageToTag, collectionFolder + "/" + str(toWriteFrom + iterationIndex) + ".png")
        makeOutput(iterationIndex, position, rotation.as_rotvec(degrees=False).tolist(), True, extraInfo={'tagLength': tagLength, 'tagFamily': 'tag36h11', 'tagId': 0})
        iterationIndex += 1

# creates DataFrame and appends it to file
collectedInfo = pd.DataFrame.from_dict({
    "imageName": imageNames,
    "arucoAvailable": arucoAvailables,
    "realT": positions,
    "realR": rotations,
    "otherInfo": otherInfos
})
collectedInfo.to_csv(collectionFolder + "/" + csvName, header=True, mode='w', index=False)