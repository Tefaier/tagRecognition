import numpy as np

tagImagesFolder = "tagImages"
collectionFolder = "collectedInfo"
resultFolder = "processedInfo"
# what this file is about
# fields: imageName, arucoAvailable, realT, realR, otherInfo
csvName = "info.csv"
# what this file is about
# fields: imageName, arucoAvailable, method, realT, realR, detectedT, detectedR
detectionFile = "detection.csv"
# what this file is about
# fields: imageName, method, isSuccess
analiseFile = "analysis.csv"

# in ratio of transform difference to correct transform
acceptedTransformError = 0.01
# in pure degrees
acceptedRotationError = 0.01
tagLength = 0.0525
imageWidth = 1280
imageHeight = 720
camMatrix=np.array(
        [[900.0, 0.0, imageWidth / 2.0],
         [0.0, 900.0, imageHeight / 2.0],
         [0.0, 0.0, 1.0]])
distortionCoefficients=np.array([0.0, 0.0, 0.0, 0.0, 0.0])