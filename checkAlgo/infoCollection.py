import os
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

tagImage = tagImagesFolder + '/' + 'aruco_1.png'
renderer = PlaneRenderer(imageWidth, imageHeight, camMatrix, tagImage)

def getImageWithParams(transform: list, rotation: Rotation, tagSize: float, saveDestination: str):
    renderer.renderPlane(transform, rotation, tagSize, saveDestination)

def makeOutput(index: int, transform: list, rotation: list, isAruco: bool = False, extraInfo: dict = None):
    if extraInfo is None:
        extraInfo = {}
    # fill values
    imageNames.append(toWriteFrom + index)
    arucoAvailables.append(isAruco)
    transforms.append(transform)
    rotations.append(rotation)
    otherInfos.append(extraInfo)

for i in range(0, 1):
    transform = [0.0, 0.0, 1.0]
    rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    getImageWithParams(transform, rotation, tagLength, collectionFolder + "/" + str(toWriteFrom + iterationIndex) + ".png")
    makeOutput(iterationIndex, transform, rotation.as_matrix().tolist(), True, extraInfo={'tagLength': tagLength, 'tagFamily': 'aruco5x5', 'tagId': 2})
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