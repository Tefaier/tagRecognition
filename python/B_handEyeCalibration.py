import json
import os.path
import time
from random import Random

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

def prepareFolder(path: str):
    ensureFolderExists(path)
    files = glob.glob(f"{path}/*")
    for f in files:
        os.remove(f)

def performEyeHand(profile: str, detector: TagDetector, generator: ImageGenerator) -> (list, list):
    prepareFolder(f"{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}/{calibrationImagesFolder}")

    # position around which images are created
    index = 0
    baseTranslation = np.array([0, 0, 0.15])
    baseRotation = Rotation.from_rotvec([180, 0, 0], degrees=True)
    positionSamples = 5
    rotationSamples = 10
    deviationTranslation = np.array([0.08, 0.02, 0.05])  # in meters
    angleRotation = 50  # in degrees

    translationsFromBase = []
    rotationsFromBase = []
    for _ in range(0, positionSamples):
        translation = baseTranslation + generateRandomNormVector() * deviationTranslation
        for _ in range(0, rotationSamples):
            rotation = baseRotation * Rotation.from_rotvec(generateRandomNormVector() * angleRotation, degrees=True)
            translationsFromBase.append(translation)
            rotationsFromBase.append(rotation.as_rotvec(degrees=False))
            generator.makeImageWithPlane(translation, rotation, f'{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}/{calibrationImagesFolder}/{index}.png')
            index += 1

    number = len(translationsFromBase)
    detectedMask, translationsFromCamera, rotationsFromCamera = performEyeHandDetection(profile, detector, number)
    translationsFromBase = np.array(translationsFromBase)[detectedMask]
    rotationsFromBase = np.array(rotationsFromBase)[detectedMask]
    rotationsFromBaseReverse = [Rotation.from_rotvec(rot, degrees=False).inv() for rot in rotationsFromBase]
    translationsFromBaseReverse = [-1 * rotationsFromBaseReverse[index].apply(tr) for index, tr in enumerate(translationsFromBase)]
    rotationsFromBaseReverse = [rot.as_rotvec(degrees=False) for rot in rotationsFromBaseReverse]
    rotationOfCamera, translationOfCamera = cv2.calibrateHandEye(rotationsFromBaseReverse, translationsFromBaseReverse, rotationsFromCamera, translationsFromCamera,
                                            method=cv2.CALIB_HAND_EYE_PARK)
    translationOfCamera = translationOfCamera.reshape((3,)).tolist()
    rotationOfCamera = Rotation.from_matrix(rotationOfCamera).as_rotvec(degrees=False).tolist()
    updateJSON({"cameraTranslation": translationOfCamera, "cameraRotation": rotationOfCamera},
               f'{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}/{generalInfoFilename}.json')
    return translationOfCamera, rotationOfCamera

def performEyeHandDetection(profile: str, detector: TagDetector, number: int) -> (list, list, list):
    translationsFromCamera = []
    rotationsFromCamera = []
    detectedMask = [True] * number

    for i in range(0, number):
        img = cv2.imread(f'{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}/{calibrationImagesFolder}/{i}.png')
        tvec, rvec, ids = detector.detect(img)
        if len(rvec) == 0:
            detectedMask[i] = False
            continue
        tvec = tvec[0]
        rvec = rvec[0]
        translationsFromCamera.append(tvec)
        rotationsFromCamera.append(rvec)

    return detectedMask, translationsFromCamera, rotationsFromCamera

def testRun():
    with open(f'{os.path.dirname(__file__)}/{generatedInfoFolder}/test/{generalInfoFilename}.json', 'r') as f:
        info: dict = json.load(f)
    chessboardPattern = (8, 6)
    patternWidth = 0.1
    patternHeight = 0.1 * 9 / 11
    squareSize = patternWidth / 11
    # performCalibrationOnExistingImages("test", patternWidth, ChessboardDetector(None, None, chessboardPattern, squareSize))
    # performEyeHand(
    #     "test",
    #     ChessboardDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")), chessboardPattern, squareSize),
    #     VTKGenerator(imageWidth, imageHeight, f'{os.path.dirname(__file__)}/{tagImagesFolder}/chessboard.png', np.array(testCameraMatrix), patternWidth,
    #                  patternHeight)
    # )
    performEyeHand(
        "test",
        ArucoDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")), patternWidth, cv2.aruco.DetectorParameters(), cv2.aruco.DICT_5X5_50),
        VTKGenerator(imageWidth, imageHeight, f'{os.path.dirname(__file__)}/{tagImagesFolder}/aruco_1.png', testCameraMatrix, patternWidth, patternWidth)
    )
