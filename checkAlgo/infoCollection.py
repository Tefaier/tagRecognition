import os

import cv2
import pandas as pd

from checkAlgo.constantsForCheck import folderName, csvName

# fields: imageName, tagFamily, tagId, transform, rotation
imageNames = []
tagFamilies = []
tagIds = []
transforms = []
rotations = []

files = os.listdir(folderName + "/")
files = list(filter(lambda name: name.split('.')[-1]=='png', files))
files = [int(name.split('.')[0]) for name in files]
toWriteFrom = max(files) + 1
iterationIndex = 0
for i in range(0, 0):
    image = cv2.imread("")
    # write image
    cv2.imwrite(filename=(folderName + "/" + str(toWriteFrom + iterationIndex) + ".png"), img=image)
    # fill values
    imageNames.append(toWriteFrom + iterationIndex)
    tagFamilies.append("aruco")
    tagIds.append("1")
    transforms.append([0, 0, 0])
    rotations.append([0, 0, 0])
    iterationIndex += 1

# creates DataFrame and appends it to file
collectedInfo = pd.DataFrame.from_dict({
    "imageName": imageNames,
    "tagFamily": tagFamilies,
    "tagId": tagIds,
    "transform": transforms,
    "rotation": rotations
})
collectedInfo.to_csv(folderName + "/" + csvName, header=False, mode='a')