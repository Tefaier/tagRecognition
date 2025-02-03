import glob
import os

import cv2.aruco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from python.models.imageGenerators.imageGenerator import ImageGenerator
from python.models.imageGenerators.vtkGenerator import VTKGenerator
from python.settings import generatedInfoFolder, analyseImagesFolder, imageInfoFilename, \
    tagImagesFolder, imageHeight, imageWidth, testCameraMatrix
from python.utils import deviateTransform, generateNormalDistributionValue, ensureFolderExists, updateJSON, \
    writeInfoToProfileJSON


class ImageGenerationSettings:
    clearExistingImages: bool
    tagLength: float
    isAruco: bool
    arucoFamily: str
    isApriltag: bool
    apriltagFamily: str

    def __init__(
            self,
            clearExistingImages: bool,
            tagLength: float,
            isAruco: bool,
            arucoFamily: str,
            isApriltag: bool,
            apriltagFamily: str
    ):
        self.clearExistingImages = clearExistingImages
        self.tagLength = tagLength
        self.isAruco = isAruco
        self.arucoFamily = arucoFamily
        self.isApriltag = isApriltag
        self.apriltagFamily = apriltagFamily

    def dictVersion(self) -> dict:
        return {"tagLength": self.tagLength, "isAruco": self.isAruco, "arucoFamily": self.arucoFamily, "isApriltag": self.isApriltag, "apriltagFamily": self.apriltagFamily}


def makeOutput(imageNames: list, name: str, translations: list, translation: list, rotations: list, rotation: list):
    imageNames.append(f"{name}.png")
    translations.append([float(val) for val in translation])
    rotations.append([float(val) for val in rotation])

def saveProfileInfo(profile: str, settings: ImageGenerationSettings):
    writeInfoToProfileJSON(profile, settings.dictVersion())

def saveGeneratedInfo(path: str, imageNames: list, translations: list, rotations: list, replaceInfo: bool):
    collectedInfo = pd.DataFrame.from_dict({
        "imageName": imageNames,
        "realT": translations,
        "realR": rotations
    })
    if replaceInfo or not os.path.exists(path):
        collectedInfo.to_csv(path, header=True, mode='w', index=False)
        return
    df = pd.read_csv(path)
    pd.concat([df, collectedInfo]).to_csv(path, header=True, mode='w', index=False)

def prepareFolder(path: str, clear: bool) -> int:
    ensureFolderExists(path)
    files = glob.glob(f"{path}/*.png")
    if clear:
        for f in files:
            os.remove(f)
        return 0
    files = [int(name.split('.')[0]) for name in files]
    toWriteFrom = max(files, default=-1) + 1
    return toWriteFrom

def generateImages(profile: str, generator: ImageGenerator, settings: ImageGenerationSettings, translations: list[list], rotations: list[Rotation]):
    toWriteFrom = prepareFolder(f"{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}/{analyseImagesFolder}", settings.clearExistingImages)
    saveProfileInfo(profile, settings)

    imageNames = []
    translationsWrite = []
    rotationsWrite = []

    p_bar = tqdm(range(len(translations)), ncols=100)

    for iterationIndex in range(len(translations)):
        generator.makeImageWithPlane(
            translations[iterationIndex],
            rotations[iterationIndex],
            f"{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}/{analyseImagesFolder}/{toWriteFrom + iterationIndex}.png"
        )
        makeOutput(
            imageNames,
            toWriteFrom + iterationIndex,
            translationsWrite,
            translations[iterationIndex],
            rotationsWrite,
            rotations[iterationIndex].as_rotvec(degrees=False).tolist()
        )
        p_bar.update()
        p_bar.refresh()

    saveGeneratedInfo(
        f"{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}/{imageInfoFilename}.csv",
        imageNames,
        translationsWrite,
        rotationsWrite,
        settings.clearExistingImages
    )


def testRun():
    translations = []
    rotations = []
    defaultTranslation = [0.0, 0.0, 4.0]
    samplesToGet = 50

    startStop, spots = (-85, 85), 50
    for x in np.linspace(startStop[0], startStop[1], spots):
        deviateValue = (startStop[1] - startStop[0]) / (spots * 2)
        rawTranslation = defaultTranslation
        rawRotation = [x + 180, 0, 0]
        for i in range(0, samplesToGet):
            translation, rotationEuler = deviateTransform(rawTranslation, rawRotation,
                                                          rx=generateNormalDistributionValue(maxDeviation=deviateValue))
            rotation = Rotation.from_euler('xyz', rotationEuler, degrees=True)
            translations.append(translation)
            rotations.append(rotation)

    startStop, spots = (-85, 85), 50
    for y in np.linspace(startStop[0], startStop[1], spots):
        deviateValue = (startStop[1] - startStop[0]) / (spots * 2)
        rawTranslation = defaultTranslation
        rawRotation = [180, y, 0]
        for i in range(0, samplesToGet):
            translation, rotationEuler = deviateTransform(rawTranslation, rawRotation,
                                                          ry=generateNormalDistributionValue(maxDeviation=deviateValue))
            rotation = Rotation.from_euler('xyz', rotationEuler, degrees=True)
            translations.append(translation)
            rotations.append(rotation)

    generateImages(
        "test",
        VTKGenerator(
            imageWidth,
            imageHeight,
            f'{os.path.dirname(__file__)}/{tagImagesFolder}/aruco_1.png',
            testCameraMatrix,
            0.1,
            0.1),
        ImageGenerationSettings(True, 0.1, True, str(cv2.aruco.DICT_5X5_100), False, ""),
        translations,
        rotations
    )
