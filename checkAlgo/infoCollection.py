import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from checkAlgo.constantsForCheck import collectionFolder, csvName, tagLength, tagImagesFolder, imageWidth, imageHeight, camMatrix
from checkAlgo.utils import deviateTransform, generateNormalDistributionValue
from checkAlgo.virtualCamera import PlaneRenderer

# fields: imageName, tagFamily, tagId, translation, rotation
imageNames = []
arucoAvailables = []
translations = []
rotations = []
otherInfos = []

files = os.listdir(collectionFolder + "/")
files = list(filter(lambda name: name.split('.')[-1]=='png', files))
files = [int(name.split('.')[0]) for name in files]
toWriteFrom = max(files, default=-1) + 1
iterationIndex = 0

tagImage = tagImagesFolder + '/' + 'aruco_1.png'
ratioOfImageToTag = 10 / 8
renderer = PlaneRenderer(imageWidth, imageHeight, camMatrix, tagImage)

def getImageWithParams(translation: list, rotation: Rotation, planeSize: float, saveDestination: str):
    renderer.renderPlane(translation, rotation, planeSize, saveDestination)

def makeOutput(index: int, translation: list, rotation: list, isAruco: bool = False, extraInfo: dict = None):
    if extraInfo is None:
        extraInfo = {}
    # fill values
    imageNames.append(str(toWriteFrom + index) + str(".png"))
    arucoAvailables.append(isAruco)
    translations.append([float(val) for val in translation])
    rotations.append(rotation)
    otherInfos.append(extraInfo)

def rotationWithRectify(toMake: Rotation) -> Rotation:
    rectify = Rotation.from_euler('xyz', [180, 0, 0], degrees=True)
    return toMake * rectify

defaultTranslation = [0.0, 0.0, 0.15]
samplesToGet = 50

for x in np.linspace(-85, 85, 60):
    deviateValue =  85  / 60
    rawTranslation = defaultTranslation
    rawRotation = [x, 0, 0]
    for i in range(0, samplesToGet):
        translation, rotationEuler = deviateTransform(rawTranslation, rawRotation, rx=generateNormalDistributionValue(maxDeviation=deviateValue))
        rotation = Rotation.from_euler('xyz', rotationEuler, degrees=True)
        getImageWithParams(translation, rotationWithRectify(rotation), tagLength * ratioOfImageToTag, collectionFolder + "/" + str(toWriteFrom + iterationIndex) + ".png")
        makeOutput(iterationIndex, translation, rotation.as_rotvec(degrees=False).tolist(), True, extraInfo={'tagLength': tagLength, 'tagFamily': 'tag36h11', 'tagId': 0})
        iterationIndex += 1
        print(f"Iteration {iterationIndex} finished")

for y in np.linspace(-85, 85, 60):
    deviateValue = 85 * 2 / (60 * 2)
    rawTranslation = defaultTranslation
    rawRotation = [0, y, 0]
    for i in range(0, samplesToGet):
        translation, rotationEuler = deviateTransform(rawTranslation, rawRotation, ry=generateNormalDistributionValue(maxDeviation=deviateValue))
        rotation = Rotation.from_euler('xyz', rotationEuler, degrees=True)
        getImageWithParams(translation, rotationWithRectify(rotation), tagLength * ratioOfImageToTag, collectionFolder + "/" + str(toWriteFrom + iterationIndex) + ".png")
        makeOutput(iterationIndex, translation, rotation.as_rotvec(degrees=False).tolist(), True, extraInfo={'tagLength': tagLength, 'tagFamily': 'tag36h11', 'tagId': 0})
        iterationIndex += 1
        print(f"Iteration {iterationIndex} finished")

for posZ in np.linspace(0.1, 4.5, 50):
    deviateValue = 4.4 / (50 * 2)
    rawTranslation = [defaultTranslation[0], defaultTranslation[1], posZ]
    rawRotation = [0, 0, 0]
    for i in range(0, samplesToGet):
        translation, rotationEuler = deviateTransform(rawTranslation, rawRotation, pz=generateNormalDistributionValue(maxDeviation=deviateValue))
        rotation = Rotation.from_euler('xyz', rotationEuler, degrees=True)
        getImageWithParams(translation, rotationWithRectify(rotation), tagLength * ratioOfImageToTag, collectionFolder + "/" + str(toWriteFrom + iterationIndex) + ".png")
        makeOutput(iterationIndex, translation, rotation.as_rotvec(degrees=False).tolist(), True, extraInfo={'tagLength': tagLength, 'tagFamily': 'tag36h11', 'tagId': 0})
        iterationIndex += 1
        print(f"Iteration {iterationIndex} finished")

for posY in np.linspace(-0.4, 0.4, 50):
    deviateValue = 0.8 / (50 * 2)
    rawTranslation = [defaultTranslation[0], posY, 1]
    rawRotation = [0, 0, 0]
    for i in range(0, samplesToGet):
        translation, rotationEuler = deviateTransform(rawTranslation, rawRotation, py=generateNormalDistributionValue(maxDeviation=deviateValue))
        rotation = Rotation.from_euler('xyz', rotationEuler, degrees=True)
        getImageWithParams(translation, rotationWithRectify(rotation), tagLength * ratioOfImageToTag, collectionFolder + "/" + str(toWriteFrom + iterationIndex) + ".png")
        makeOutput(iterationIndex, translation, rotation.as_rotvec(degrees=False).tolist(), True, extraInfo={'tagLength': tagLength, 'tagFamily': 'tag36h11', 'tagId': 0})
        iterationIndex += 1
        print(f"Iteration {iterationIndex} finished")

# creates DataFrame and appends it to file
collectedInfo = pd.DataFrame.from_dict({
    "imageName": imageNames,
    "arucoAvailable": arucoAvailables,
    "realT": translations,
    "realR": rotations,
    "otherInfo": otherInfos
})
collectedInfo.to_csv(collectionFolder + "/" + csvName, header=True, mode='w', index=False)