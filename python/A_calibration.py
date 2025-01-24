import os
import os.path

import numpy as np
import cv2
import glob

from scipy.spatial.transform import Rotation

from python.models.detectors.arucoDetector import ArucoDetector
from python.models.detectors.chessboardDetector import ChessboardDetector
from python.models.detectors.detector import TagDetector
from python.models.imageGenerators.imageGenerator import ImageGenerator
from python.models.imageGenerators.vtkGenerator import VTKGenerator
from python.settings import generatedInfoFolder, calibrationImagesFolder, imageWidth, imageHeight, tagImagesFolder, \
    testCameraMatrix, generalInfoFilename
from python.utils import ensureFolderExists, getGrayImage, generateRandomNormVector, updateJSON


def performCalibration(profile: str, tagLength: float, detector: TagDetector, generator: ImageGenerator) -> (list, list):
    ensureFolderExists(f"{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}/{calibrationImagesFolder}")
    files = glob.glob(f"{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}/{calibrationImagesFolder}/*")
    for f in files:
        os.remove(f)

    # position around which images are created
    index = 0
    baseTranslation = np.array([0, 0, 0.15])
    baseRotation = Rotation.from_rotvec([180, 0, 0], degrees=True)
    positionSamples = 5
    rotationSamples = 10
    deviationTranslation = np.array([0.08, 0.02, 0.05])  # in meters
    angleRotation = 50  # in degrees
    for _ in range(0, positionSamples):
        translation = baseTranslation + generateRandomNormVector() * deviationTranslation
        for _ in range(0, rotationSamples):
            rotation = baseRotation * Rotation.from_rotvec(generateRandomNormVector() * angleRotation, degrees=True)
            generator.makeImageWithPlane(translation, rotation, f'{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}/{calibrationImagesFolder}/{index}.png')
            index += 1

    cameraMatrix, distortionCoefficients = performCalibrationOnExistingImages(profile, tagLength, detector)
    return cameraMatrix, distortionCoefficients

def performCalibrationOnExistingImages(profile: str, tagLength: float, detector: TagDetector) -> (list, list):
    ensureFolderExists(f"{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}/{calibrationImagesFolder}")
    objpoints = []
    imgpoints = []
    images = glob.glob(f'{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}/{calibrationImagesFolder}/*.png')

    for name in images:
        image = cv2.imread(name)
        objp, imgp = detector.detectObjectPoints(image, tagLength)
        if imgp is not None:
            objpoints.append(objp)
            imgpoints.append(imgp)

    ret, cameraMatrix, distortionCoefficients, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, getGrayImage(cv2.imread(images[-1])).shape[::-1], None, None, flags = cv2.CALIB_USE_LU)
    cameraMatrix = cameraMatrix.reshape((4, 4)).tolist()
    distortionCoefficients = distortionCoefficients.reshape((5,)).tolist()
    updateJSON({"cameraMatrix": cameraMatrix, "distortionCoefficients": distortionCoefficients},
               f'{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}/{generalInfoFilename}.json')
    return cameraMatrix, distortionCoefficients

def testRun():
    chessboardPattern = (8, 6)
    patternWidth = 0.1
    patternHeight = 0.1 * 9 / 11
    squareSize = patternWidth / 11
    # performCalibrationOnExistingImages("test", patternWidth, ChessboardDetector(None, None, chessboardPattern, squareSize))
    performCalibration(
        "test", patternWidth,
        ChessboardDetector(None, None, chessboardPattern, squareSize),
        VTKGenerator(imageWidth, imageHeight, f'{os.path.dirname(__file__)}/{tagImagesFolder}/chessboard.png', testCameraMatrix, patternWidth,
                     patternHeight)
    )
    # performCalibration(
    #     "test", patternWidth,
    #     ArucoDetector(None, None, cv2.aruco.DICT_5X5_50),
    #     VTKGenerator(imageWidth, imageHeight, f'{os.path.dirname(__file__)}/{tagImagesFolder}/aruco_1.png', testCameraMatrix, patternWidth,
    #                  patternWidth)
    # )
