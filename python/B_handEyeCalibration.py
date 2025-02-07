import json
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
from python.models.transformsParser.transformsParser import TransformsParser
from python.settings import generated_info_folder, calibration_images_folder, image_width, image_height, tag_images_folder, \
    test_camera_matrix, general_info_filename
from python.utils import ensure_folder_exists, generate_random_norm_vector, write_info_to_profile_json

def prepare_folder(path: str):
    ensure_folder_exists(path)
    files = glob.glob(f"{path}/*")
    for f in files:
        os.remove(f)

def perform_eye_hand(profile: str, detector: TagDetector, parser: TransformsParser, generator: ImageGenerator) -> (list, list):
    prepare_folder(f"{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{calibration_images_folder}")

    # position around which images are created
    index = 0
    base_translation = np.array([0, 0, 0.15])
    base_rotation = Rotation.from_rotvec([180, 0, 0], degrees=True)
    position_samples = 5
    rotation_samples = 10
    deviation_translation = np.array([0.08, 0.02, 0.05])  # in meters
    angle_rotation = 50  # in degrees

    translations_from_base = []
    rotations_from_base = []
    for _ in range(0, position_samples):
        translation = base_translation + generate_random_norm_vector() * deviation_translation
        for _ in range(0, rotation_samples):
            rotation = base_rotation * Rotation.from_rotvec(generate_random_norm_vector() * angle_rotation, degrees=True)
            translations_from_base.append(translation)
            rotations_from_base.append(rotation.as_rotvec(degrees=False))
            generator.generate_image_with_obj_at_transform(translation, rotation, f'{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{calibration_images_folder}/{index}.png')
            index += 1

    number = len(translations_from_base)
    detected_mask, translations_from_camera, rotations_from_camera = perform_eye_hand_detection(profile, detector, parser, number)
    translations_from_base = np.array(translations_from_base)[detected_mask]
    rotations_from_base = np.array(rotations_from_base)[detected_mask]
    rotations_from_base_reverse = [Rotation.from_rotvec(rot, degrees=False).inv() for rot in rotations_from_base]
    translations_from_base_reverse = [-1 * rotations_from_base_reverse[index].apply(tr) for index, tr in enumerate(translations_from_base)]
    rotations_from_base_reverse = [rot.as_rotvec(degrees=False) for rot in rotations_from_base_reverse]
    cameraRotation, cameraTranslation = cv2.calibrateHandEye(rotations_from_base_reverse, translations_from_base_reverse, rotations_from_camera, translations_from_camera,
                                            method=cv2.CALIB_HAND_EYE_PARK)
    cameraTranslation = cameraTranslation.reshape((3,)).tolist()
    cameraRotation = Rotation.from_matrix(cameraRotation).as_rotvec(degrees=False).tolist()
    write_info_to_profile_json(profile, {"cameraTranslation": cameraTranslation, "cameraRotation": cameraRotation})
    return cameraTranslation, cameraRotation

def perform_eye_hand_detection(profile: str, detector: TagDetector, parser: TransformsParser, number: int) -> (list, list, list):
    translations_from_camera = []
    rotations_from_camera = []
    detected_mask = [True] * number

    for i in range(0, number):
        img = cv2.imread(f'{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{calibration_images_folder}/{i}.png')
        tvec, rvec, ids = detector.detect(img)
        tvec, rvec = parser.get_parent_transform(tvec, [Rotation.from_rotvec(rotation, degrees=False) for rotation in rvec], ids)
        if len(rvec) == 0:
            detected_mask[i] = False
            continue
        translations_from_camera.append(tvec)
        rotations_from_camera.append(rvec)

    return detected_mask, translations_from_camera, rotations_from_camera

def test_run():
    with open(f'{os.path.dirname(__file__)}/{generated_info_folder}/test/{general_info_filename}.json', 'r') as f:
        info: dict = json.load(f)
    chessboard_pattern = (8, 6)
    pattern_width = 0.1
    pattern_height = 0.1 * 9 / 11
    square_size = pattern_width / 11
    # performCalibrationOnExistingImages("test", pattern_width, ChessboardDetector(None, None, chessboard_pattern, square_size))
    # performEyeHand(
    #     "test",
    #     ChessboardDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")), chessboard_pattern, square_size),
    #     VTKGenerator(imageWidth, imageHeight, f'{os.path.dirname(__file__)}/{tagImagesFolder}/chessboard.png', np.array(testCameraMatrix), pattern_width,
    #                  pattern_height)
    # )
    perform_eye_hand(
        "test",
        ArucoDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")), pattern_width, cv2.aruco.DetectorParameters(), cv2.aruco.DICT_5X5_50),
        TransformsParser([[0, 0, 0]], [Rotation.from_rotvec([0, 0, 0])], [2]),
        VTKGenerator(image_width, image_height, [np.array([0, 0, 0])], [Rotation.from_rotvec([0, 0, 0])],[f'{os.path.dirname(__file__)}/{tag_images_folder}/aruco_1.png'], test_camera_matrix, pattern_width, pattern_width)
    )
