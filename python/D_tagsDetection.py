import json
import os

import cv2
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from python.models.detectors.arucoDetector import ArucoDetector
from python.models.detectors.detector import TagDetector
from python.models.transformsParser.transformsParser import TransformsParser
from python.settings import generatedInfoFolder, imageInfoFilename, detectionInfoFilename, analyseImagesFolder, \
    generalInfoFilename
from python.utils import parseRotation, readStringOfList


def openAndPrepareRawInfo(path: str) -> pd.DataFrame:
    info = pd.read_csv(path)
    # info = info.reset_index()
    return info

def getVectorError(vector1: list, vector2: list) -> list:
    if (len(vector1) == 0 or len(vector2) == 0): return []
    return [vector2[i] - vector1[i] for i in range(0, len(vector1))]


def getRotationError(rotation1: list, rotation2: list) -> list:
    rotation1 = parseRotation(rotation1)
    rotation2 = parseRotation(rotation2)
    if (rotation1 == None or rotation2 == None): return []

    rotation1To2 = rotation2 * rotation1.inv()
    return rotation1To2.as_rotvec(degrees=False).tolist()

def analyseInfo(imagesFolder: str, detector: TagDetector, dframe: pd.DataFrame, parser: TransformsParser):
    detectorName = np.full((dframe.shape[0],), detector.name)
    detectedT = []
    detectedR = []

    p_bar = tqdm(range(dframe.shape[0]), ncols=100)
    writeDetectionInfo(p_bar, imagesFolder, detector, dframe, detectedT, detectedR, parser)
    p_bar.close()
    dframe["method"] = detectorName
    dframe["detectedT"] = detectedT
    dframe["detectedR"] = detectedR
    p_bar = tqdm(range(dframe.shape[0]), ncols=100)
    writeErrorInfo(p_bar, dframe)

def writeDetectionInfo(bar: tqdm, imagesFolder: str, detector: TagDetector, dframe: pd.DataFrame, translationWrite: list, rotationWrite: list, parser: TransformsParser):
    for _, row in dframe.iterrows():
        t, r, ids = detector.detect(image=cv2.imread(f"{imagesFolder}/{row["imageName"]}"))
        t, r = parser.getParentTransform(t, [Rotation.from_rotvec(rotation, degrees=False) for rotation in r], ids)
        translationWrite.append([float(val) for val in t])
        rotationWrite.append([float(val) for val in r])
        bar.update()
        bar.refresh()

def writeErrorInfo(bar: tqdm, dframe: pd.DataFrame):
    realT = readStringOfList(dframe['realT'])
    realR = readStringOfList(dframe['realR'])
    detectedT = dframe['detectedT']
    detectedR = dframe['detectedR']

    errorT = []
    errorR = []
    isSuccess = np.full((dframe.shape[0],), False)
    for i in range(0, dframe.shape[0]):
        errorT.append(getVectorError(realT[i], detectedT[i]))
        errorR.append(getRotationError(realR[i], detectedR[i]))
        isSuccess[i] = len(errorT[-1]) != 0 and len(errorR[-1]) != 0
        bar.update()
        bar.refresh()

    dframe['isSuccess'] = isSuccess
    dframe['errorT'] = errorT
    dframe['errorR'] = errorR

def writeInfoToFile(path: str, dframe: pd.DataFrame, detectionSettings: dict, replace: bool):
    dframe["detectionSettings"] = np.full((dframe.shape[0],), detectionSettings)

    if replace or not os.path.exists(path):
        dframe.to_csv(path, header=True, mode='w', index=False)
        return
    df = pd.read_csv(path)
    pd.concat([df, dframe]).to_csv(path, header=True, mode='w', index=False)

def performDetection(profile: str, detector: TagDetector, parser: TransformsParser, replaceInfo: bool):
    profilePath = f"{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}"
    imagesInfo = openAndPrepareRawInfo(f"{profilePath}/{imageInfoFilename}.csv")
    analyseInfo(f"{profilePath}/{analyseImagesFolder}", detector, imagesInfo, parser)
    writeInfoToFile(
        f"{profilePath}/{detectionInfoFilename}.csv",
        imagesInfo,
        detector.detectorSettings(),
        replaceInfo
    )

def testRun():
    with open(f'{os.path.dirname(__file__)}/{generatedInfoFolder}/test/{generalInfoFilename}.json', 'r') as f:
        info: dict = json.load(f)

    performDetection("test", ArucoDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")), info["tagLength"], cv2.aruco.DetectorParameters(), int(info["arucoFamily"])), TransformsParser([[0, 0, 0]], [Rotation.from_rotvec([0, 0, 0])], [2]), True)
