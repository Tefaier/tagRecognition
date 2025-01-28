import numpy as np

generatedInfoFolder = 'generatedInfo'
calibrationImagesFolder = 'calibration'
analyseImagesFolder = 'analyse'
generalInfoFilename = 'profileInfo'
imageInfoFilename = 'imageInfo'
detectionInfoFilename = 'detectionInfo'
plotsFolder = 'plots'

tagImagesFolder = 'tagImages'

imageWidth = 1920
imageHeight = 1280

testCameraMatrix=np.array(
        [[1000.0, 0.0, imageWidth / 2.0],
         [0.0, 1000.0, imageHeight / 2.0],
         [0.0, 0.0, 1.0]])
testDistortionCoefficients=np.array([0.0, 0.0, 0.0, 0.0, 0.0])