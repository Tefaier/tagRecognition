import time
from random import Random

import numpy as np
import cv2
import glob

from scipy.spatial.transform import Rotation

from python.models.detectors.chessboardDetector import ChessboardDetector
from python.models.detectors.detector import TagDetector
from python.models.imageGenerators.imageGenerator import ImageGenerator
from python.models.imageGenerators.vtkGenerator import VTKGenerator
from python.settings import generatedInfoFolder, calibrationImagesFolder, imageWidth, imageHeight, tagImagesFolder, \
    testCameraMatrix
from python.utils import ensureFolderExists, getGrayImage, generateRandomNormVector


def performHandEye(profile: str, detector: TagDetector, generator: ImageGenerator) -> (list, list):
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

    return performHandEyeOnExistingImages(profile, detector)

def performHandEyeOnExistingImages(profile: str, detector: TagDetector) -> (list, list):
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
    return cameraMatrix, distortionCoefficients

if __name__ == "__main__":
    performHandEye(
        "test",
        ChessboardDetector(None, None, (9, 7), 0.1),
        VTKGenerator(imageWidth, imageHeight, f'{tagImagesFolder}/chessboard.png', testCameraMatrix, 0.1, 0.1)
    )