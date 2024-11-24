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
camMatrix=np.array(
        [[804.7329058535828, 0.0, 549.3237487667773],
         [0.0, 802.189566021595, 293.62680986426403],
         [0.0, 0.0, 1.0]])
distortionCoefficients=np.array([1, 1, 1, 1, 1])