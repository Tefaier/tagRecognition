import json
import os.path
from typing import Tuple

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
from python.utils import ensure_folder_exists, generate_random_norm_vector, write_info_to_profile_json, random_generator


def prepare_folder(path: str):
    ensure_folder_exists(path)
    files = glob.glob(f"{path}/*")
    for f in files:
        os.remove(f)

def perform_eye_hand(profile: str, detector: TagDetector, parser: TransformsParser, generator: ImageGenerator, distance_range: Tuple[float, float], x_deviation_angle: float, y_deviation_angle: float, obj_rotation_limit: float, rotate_from: Rotation) -> (list, list):
    prepare_folder(f"{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{calibration_images_folder}")

    # position around which images are created
    index = 0
    position_samples = 5
    rotation_samples = 10

    translations_from_base = []
    rotations_from_base = []
    for _ in range(0, position_samples):
        translation = Rotation.from_euler('xyz', [random_generator.uniform(-x_deviation_angle, x_deviation_angle), random_generator.uniform(-y_deviation_angle, y_deviation_angle), 0], degrees=True).apply(np.array([0, 0, random_generator.uniform(distance_range[0], distance_range[1])]))
        for _ in range(0, rotation_samples):
            rotation = Rotation.from_rotvec(generate_random_norm_vector() * obj_rotation_limit, degrees=True) * rotate_from
            success = generator.generate_image_with_obj_at_transform(translation, rotation, f'{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{calibration_images_folder}/{index}.png')
            if success:
                translations_from_base.append(translation)
                rotations_from_base.append(rotation.as_rotvec(degrees=False))
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
        VTKGenerator(image_width, image_height, [np.array([0, 0, 0])], [Rotation.from_rotvec([0, 0, 0])],[f'{os.path.dirname(__file__)}/{tag_images_folder}/aruco_1.png'], test_camera_matrix, pattern_width * 450 / 354, pattern_width * 450 / 354),
        (0.2, 0.3), 20, 10, 60, Rotation.from_rotvec([180, 0, 0], degrees=True)
    )
