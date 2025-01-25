import json
import os

import cv2
import pandas as pd
import numpy as np

from python.models.detectors.arucoDetector import ArucoDetector
from python.models.detectors.detector import TagDetector
from python.models.transformsParser.simpleParser import SimpleParser
from python.models.transformsParser.transformsParser import TransformsParser
from python.settings import generatedInfoFolder, imageInfoFilename, detectionInfoFilename, analyseImagesFolder, \
    generalInfoFilename

def openAndPrepareRawInfo(path: str) -> pd.DataFrame:
    info = pd.read_csv(path)
    # info = info.reset_index()
    return info

def analyseInfo(imagesFolder: str, detector: TagDetector, dframe: pd.DataFrame, parser: TransformsParser):
    detectorName = np.full((dframe.shape[0],), detector.name)
    detectedT = []
    detectedR = []

    performDetection(imagesFolder, detector, dframe, detectedT, detectedR, parser)
    dframe["method"] = detectorName
    dframe["detectedT"] = detectedT
    dframe["detectedR"] = detectedR

def performDetection(imagesFolder: str, detector: TagDetector, dframe: pd.DataFrame, translationWrite: list, rotationWrite: list, parser: TransformsParser):
    for _, row in dframe.iterrows():
        t, r, ids = detector.detect(image=cv2.imread(f"{imagesFolder}/{row["imageName"]}"))
        t, r = parser.getParentTransform(t, r, ids)
        translationWrite.append([float(val) for val in t])
        rotationWrite.append([float(val) for val in r])

def writeInfo(path: str, dframe: pd.DataFrame, detectionSettings: dict, replace: bool):
    dframe["detectionSettings"] = np.full((dframe.shape[0],), detectionSettings)

    if replace or not os.path.exists(path):
        dframe.to_csv(path, header=True, mode='w', index=False)
        return
    df = pd.read_csv(path)
    pd.concat([df, dframe]).to_csv(path, header=True, mode='w', index=False)

def performDetection(profile: str, detector: TagDetector, detectionSettings: dict, parser: TransformsParser, replaceInfo: bool):
    profilePath = f"{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}"
    imagesInfo = openAndPrepareRawInfo(f"{profilePath}/{imageInfoFilename}.csv")
    analyseInfo(f"{profilePath}/{analyseImagesFolder}", detector, imagesInfo, parser)
    writeInfo(
        f"{profilePath}/{detectionInfoFilename}.csv",
        imagesInfo,
        detectionSettings,
        replaceInfo
    )

def testRun():
    with open(f'{os.path.dirname(__file__)}/{generatedInfoFolder}/test/{generalInfoFilename}.json', 'r') as f:
        info: dict = json.load(f)

    performDetection("test", ArucoDetector(info["cameraMatrix"], info["distortionCoefficients"], info["tagLength"], int(info["arucoFamily"])), {}, SimpleParser([[0, 0, 0]], [[0, 0, 0]], [5]), True)
