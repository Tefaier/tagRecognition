import os

import cv2
import pandas as pd

from checkAlgo.constantsForCheck import collectionFolder, csvName

# fields: imageName, tagFamily, tagId, transform, rotation
imageNames = []
onlyArucos = []
transforms = []
rotations = []

files = os.listdir(collectionFolder + "/")
files = list(filter(lambda name: name.split('.')[-1]=='png', files))
files = [int(name.split('.')[0]) for name in files]
toWriteFrom = max(files) + 1
iterationIndex = 0
for i in range(0, 0):
    image = cv2.imread("")
    # write image
    cv2.imwrite(filename=(collectionFolder + "/" + str(toWriteFrom + iterationIndex) + ".png"), img=image)
    # fill values
    imageNames.append(toWriteFrom + iterationIndex)
    onlyArucos.append(True)
    transforms.append([0, 0, 0])
    rotations.append([0, 0, 0])
    iterationIndex += 1

# creates DataFrame and appends it to file
collectedInfo = pd.DataFrame.from_dict({
    "imageName": imageNames,
    "onlyAruco": onlyArucos,
    "transform": transforms,
    "rotation": rotations
})
collectedInfo.to_csv(collectionFolder + "/" + csvName, header=False, mode='a')