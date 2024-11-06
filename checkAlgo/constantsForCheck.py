collectionFolder = "collectedInfo"
resultFolder = "processedInfo"
# what this file is about
# fields: imageName, onlyAruko, transform, rotation
csvName = "info.csv"
# what this file is about
# fields: imageName, onlyAruko, method, realT, realR, detectedT, detectedR
detectionFile = "detection.csv"
# what this file is about
# fields: imageName, method, isSuccess
analiseFile = "analysis.csv"

# in ratio of transform difference to correct transform
acceptedTransformError = 0.01
# in pure degrees
acceptedRotationError = 0.01