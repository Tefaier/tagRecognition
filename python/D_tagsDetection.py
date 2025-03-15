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
from python.settings import generated_info_folder, image_info_filename, detection_info_filename, analyse_images_folder, \
    general_info_filename
from python.utils import parse_rotation, read_string_of_list


def _open_and_prepare_raw_info(path: str) -> pd.DataFrame:
    info = pd.read_csv(path)
    if info.get("time") is not None:
        info = info.sort_values(by="time").reset_index().drop(columns=["index"])
    # info = info.reset_index()
    return info

def _get_vector_error(vector1: list, vector2: list) -> list:
    if len(vector1) == 0 or len(vector2) == 0: return []
    return [vector2[i] - vector1[i] for i in range(0, len(vector1))]


def _get_rotation_error(rotation1: list, rotation2: list) -> list:
    rotation1 = parse_rotation(rotation1)
    rotation2 = parse_rotation(rotation2)
    if rotation1 is None or rotation2 is None: return []

    rotation1To2 = rotation2 * rotation1.inv()
    return rotation1To2.as_rotvec(degrees=False).tolist()

def _analyse_info(images_folder: str, detector: TagDetector, dframe: pd.DataFrame, parser: TransformsParser):
    detector_name = np.full((dframe.shape[0],), detector.name)
    detectedT = []
    detectedR = []

    print("Start of tags detection")
    p_bar = tqdm(range(dframe.shape[0]), ncols=100)
    _write_detection_info(p_bar, images_folder, detector, dframe, detectedT, detectedR, parser)
    p_bar.close()
    dframe["method"] = detector_name
    dframe["detectedT"] = detectedT
    dframe["detectedR"] = detectedR

    print("Start of calculating deviation")
    p_bar = tqdm(range(dframe.shape[0]), ncols=100)
    _write_error_info(p_bar, dframe)
    p_bar.close()

# in dframe time if present is expected to be in ascending order
# it is already fulfilled if dframe is opened using _open_and_prepare_raw_info
def _write_detection_info(bar: tqdm, images_folder: str, detector: TagDetector, dframe: pd.DataFrame, translation_write: list, rotation_write: list, parser: TransformsParser):
    for _, row in dframe.iterrows():
        t, r, ids = detector.detect(image=cv2.imread(f"{images_folder}/{row["imageName"]}"))
        t, r = parser.get_parent_transform(t, [Rotation.from_rotvec(rotation, degrees=False) for rotation in r], ids, row.get("time"))
        translation_write.append([float(val) for val in t])
        rotation_write.append([float(val) for val in r])
        bar.update()
        bar.refresh()

def _write_error_info(bar: tqdm, dframe: pd.DataFrame):
    realT = read_string_of_list(dframe['realT'])
    realR = read_string_of_list(dframe['realR'])
    detectedT = dframe['detectedT']
    detectedR = dframe['detectedR']

    errorT = []
    errorR = []
    isSuccess = np.full((dframe.shape[0],), False)
    for i in range(0, dframe.shape[0]):
        errorT.append(_get_vector_error(realT[i], detectedT[i]))
        errorR.append(_get_rotation_error(realR[i], detectedR[i]))
        isSuccess[i] = len(errorT[-1]) != 0 and len(errorR[-1]) != 0
        bar.update()
        bar.refresh()

    dframe['isSuccess'] = isSuccess
    dframe['errorT'] = errorT
    dframe['errorR'] = errorR

def _write_info_to_file(path: str, dframe: pd.DataFrame, detectionSettings: dict, replace: bool):
    dframe["detectionSettings"] = np.full((dframe.shape[0],), detectionSettings)

    if replace or not os.path.exists(path):
        dframe.to_csv(path, header=True, mode='w', index=False)
        return
    df = pd.read_csv(path)
    pd.concat([df, dframe]).to_csv(path, header=True, mode='w', index=False)

def perform_detection(profile: str, detector: TagDetector, parser: TransformsParser, replace_info: bool):
    profilePath = f"{os.path.dirname(__file__)}/{generated_info_folder}/{profile}"
    imagesInfo = _open_and_prepare_raw_info(f"{profilePath}/{image_info_filename}.csv")
    _analyse_info(f"{profilePath}/{analyse_images_folder}", detector, imagesInfo, parser)
    _write_info_to_file(
        f"{profilePath}/{detection_info_filename}.csv",
        imagesInfo,
        detector.detector_settings(),
        replace_info
    )

def test_run():
    with open(f'{os.path.dirname(__file__)}/{generated_info_folder}/test/{general_info_filename}.json', 'r') as f:
        info: dict = json.load(f)

    perform_detection("test", ArucoDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")), info["tagSize"], cv2.aruco.DetectorParameters(), int(info["arucoFamily"])), TransformsParser([[0, 0, 0]], [Rotation.from_rotvec([0, 0, 0])], [2]), True)
