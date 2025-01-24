import time
from random import Random

import numpy as np
import cv2
import glob
import json

from scipy.spatial.transform import Rotation

from main.models.detectors.chessboardDetector import ChessboardDetector
from main.models.detectors.detector import TagDetector
from main.models.imageGenerators.imageGenerator import ImageGenerator
from main.models.imageGenerators.vtkGenerator import VTKGenerator
from main.settings import generatedInfoFolder, calibrationImagesFolder, imageWidth, imageHeight, tagImagesFolder, \
    testCameraMatrix, generalInfoFilename
from main.utils import ensureFolderExists, getGrayImage, generateRandomNormVector


def performCalibration(profile: str, detector: TagDetector, generator: ImageGenerator) -> (list, list):
    random = Random()
    random.seed(int(time.time()))
    ensureFolderExists(f"{generatedInfoFolder}/{profile}/{calibrationImagesFolder}")

    index = 0
    # position around which images are created
    baseTranslation = np.array([0, 0, 1])
    baseRotation = Rotation.from_rotvec([180, 0, 0], degrees=True)
    positionSamples = 10
    rotationSamples = 5
    deviationTranslation = 0.5  # in meters
    angleRotation = 30  # in degrees
    for _ in range(0, positionSamples):
        translation = baseTranslation + generateRandomNormVector() * deviationTranslation
        for _ in range(0, rotationSamples):
            rotation = baseRotation * Rotation.from_rotvec(generateRandomNormVector() * angleRotation, degrees=True)
            generator.makeImageWithPlane(translation, rotation, f'{generatedInfoFolder}/{profile}/{calibrationImagesFolder}/{index}.png')
            index += 1

    cameraMatrix, distortionCoefficients = performCalibrationOnExistingImages(profile, detector)
    with open(f'{generatedInfoFolder}/{profile}/{generalInfoFilename}.json', 'w') as fp:
        json.dump({"cameraMatrix": cameraMatrix, "distortionCoefficients": distortionCoefficients}, fp)
    return cameraMatrix, distortionCoefficients

def performCalibrationOnExistingImages(profile: str, detector: TagDetector) -> (list, list):
    ensureFolderExists(f"{generatedInfoFolder}/{profile}/{calibrationImagesFolder}")
    objpoints = []
    imgpoints = []
    images = glob.glob(f'{generatedInfoFolder}/{profile}/{calibrationImagesFolder}/*.png')

    for name in images:
        image = cv2.imread(name)
        objp, imgp = detector.detectObjectPoints(image, 0)
        if imgp is not None:
            objpoints.append(objp.copy())
            imgpoints.append(imgp)

    ret, cameraMatrix, distortionCoefficients, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, getGrayImage(cv2.imread(images[0])).shape[::-1], None, None)
    cameraMatrix = cameraMatrix.tolist()
    distortionCoefficients = distortionCoefficients.tolist()
    return cameraMatrix, distortionCoefficients

if __name__ == "__main__":
    chessboardPattern = (8, 6)
    patternWidth = 0.1
    patternHeight = 0.1 * 9 / 11
    squareSize = patternWidth / 11
    performCalibration(
        "test",
        ChessboardDetector(None, None, chessboardPattern, squareSize),
        VTKGenerator(imageWidth, imageHeight, f'{tagImagesFolder}/chessboard.png', testCameraMatrix, patternWidth, patternHeight)
    )