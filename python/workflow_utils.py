import glob
import os

import cv2
import pandas as pd
from scipy.spatial.transform import Rotation

from python.A_calibration import perform_calibration
from python.B_handEyeCalibration import perform_eye_hand, _run_calibration
from python.C_imagesGeneration import ImageGenerationSettings, generate_images
from python.D_tagsDetection import perform_detection
from python.E_visualization import read_info, get_info_part
from python.experiments import x_y_experiment, x_z_experiment, x_rx_experiment, x_ry_experiment, x_rz_experiment, \
    simple_trajectory_experiment, simple_trajectory_rotation_experiment
from python.models.detectors.arucoDetector import ArucoDetector
# from python.models.detectors.apriltagDetector import ApriltagDetector, ApriltagSettings
from python.models.detectors.chessboardDetector import ChessboardDetector
from python.models.detectors.detector import TagDetector
from python.models.imageGenerators.manipulatorGenerator import ManipulatorGenerator
from python.models.imageGenerators.pseudoGenerator import PseudoImageGenerator
from python.models.imageGenerators.vtkGenerator import VTKGenerator
import numpy as np

from python.models.transformsParser.cubeParser import CubeParser
from python.models.transformsParser.kalmanParser import SimpleKalmanFilterParser
from python.models.transformsParser.transformsParser import TransformsParser
from python.settings import tag_images_folder, test_camera_matrix, generated_info_folder, image_info_filename, \
    detection_info_filename, analyse_images_folder
from python.utils import read_profile_json, change_base2gripper_to_camera2object, write_info_to_profile_json, \
    copy_camera_profile_info


# detector_type either aruco or apriltag
# transforms_type either with traj or not
def create_image_generation_settings(detector_type: str, transforms_type: str) -> ImageGenerationSettings:
    return ImageGenerationSettings(
        True,
        0.0708,
        detector_type == 'aruco',
        str(cv2.aruco.DICT_5X5_50) if detector_type == 'aruco' else '',
        detector_type == 'apriltag',
        "tag36h11" if detector_type == 'apriltag' else '',
        "traj" in transforms_type
    )

# setup_type either single or cube
def create_parser(settings: ImageGenerationSettings, setup_type: str) -> TransformsParser:
    if setup_type == "single":
        return TransformsParser([[0, 0, settings.tagSize * 450 / 354 / 2]], [Rotation.from_rotvec([0, 0, 90], degrees=True)], [2])
    else:
        return CubeParser([0, 1, 2, 3, 4, 5], settings.tagSize * 450 / 354)

# detector_type either aruco or apriltag
# setup_type either single or cube
def create_vtk_generator(
        settings: ImageGenerationSettings,
        used_transform: TransformsParser,
        detector_type: str,
        setup_type: str,
        camera_matrix: np.ndarray
) -> VTKGenerator:
    if detector_type == "aruco":
        path_creator = lambda i: f'{os.path.dirname(__file__)}/{tag_images_folder}/aruco_5x5_{i}.png'
    else:
        path_creator = lambda i: f'{os.path.dirname(__file__)}/{tag_images_folder}/april_36h11_{i}.png'
    paths = [2] if setup_type == "single" else [0, 1, 2, 3, 4, 5]
    paths = [path_creator(path) for path in paths]
    return VTKGenerator(
        1920,
        1080,
        used_transform.translations,
        used_transform.rotations,
        paths,
        camera_matrix,
        settings.tagSize * 450 / 354,
        settings.tagSize * 450 / 354
    )

def create_manipulator_generator(
        base2camera_translation: np.array,
        base2camera_rotation: Rotation
) -> ManipulatorGenerator:
    return ManipulatorGenerator(
        True,
        "192.168.55.55",
        30003,
        base2camera_translation,
        base2camera_rotation,
        np.array([0, 0, 0.107]),
        Rotation.from_rotvec([0, 0, 0]),
        2, # set 2 if Linux, 1 if Windows
        False
    )

def create_aruco_detector(profile: str, settings: ImageGenerationSettings, aruco3: bool) -> ArucoDetector:
    info = read_profile_json(profile)
    parameters = cv2.aruco.DetectorParameters()
    parameters.useAruco3Detection = aruco3
    return ArucoDetector(
        np.array(info.get("cameraMatrix")),
        np.array(info.get("distortionCoefficients")),
        settings.tagSize,
        parameters,
        cv2.aruco.DICT_5X5_50
    )

# def create_apriltag_detector(profile: str, settings: ImageGenerationSettings) -> ApriltagDetector:
#     info = read_profile_json(profile)
#     return ApriltagDetector(
#         np.array(info.get("cameraMatrix")),
#         np.array(info.get("distortionCoefficients")),
#         settings.tagSize,
#         ApriltagSettings(),
#         settings.apriltagFamily
#     )

def create_transforms(
        base2camera_translation: np.array,
        base2camera_rotation: Rotation,
        transforms_type: str,
):
    if transforms_type == "x_y":
        t, r, s = x_y_experiment(20, 5)
    elif transforms_type == "x_z":
        t, r, s = x_z_experiment(20, 5)
    elif transforms_type == "x_rx":
        t, r, s = x_rx_experiment(10, 10)
    elif transforms_type == "x_ry":
        t, r, s = x_ry_experiment(10, 10)
    elif transforms_type == "x_rz":
        t, r, s = x_rz_experiment(10, 10)
    elif transforms_type == "traj_1":
        t, r, s = simple_trajectory_experiment(15)
    elif transforms_type == "traj_2":
        t, r, s = simple_trajectory_rotation_experiment(15)
    t, r = change_base2gripper_to_camera2object(base2camera_translation, base2camera_rotation, np.array([0, 0, 0.107]), Rotation.from_rotvec([0, 0, 0]), np.array(t), r)
    return t, r, s

def create_detections(profile: str, settings: ImageGenerationSettings, parser: TransformsParser, detector_type: str, setup_type: str, transforms_type: str):
    used_detectors = []
    if detector_type == "aruco":
        for aruco3 in [False, True]:
            used_detectors.append(create_aruco_detector(profile, settings, aruco3))
    else:
        used_detectors.append(create_apriltag_detector(profile, settings))

    used_parsers = [parser]
    if "traj" in transforms_type:
        for flip in [False, True]:
            for filter in [False] if setup_type == "single" else [False, True]:
                used_parsers.append(SimpleKalmanFilterParser(
                    parser,
                    1,
                    flip,
                    filter
                ))

    iteration = 0
    for detector in used_detectors:
        for i in range(0, len(used_parsers)):
            perform_detection(profile, detector, used_parsers[i], iteration == 0)
            iteration += 1

def camera_calibration(profile: str, is_virtual: bool, base2camera_translation: np.array = None, base2camera_rotation: Rotation = None):
    if is_virtual:
        square_size = 0.1 / 11
        perform_calibration(
            profile,
            ChessboardDetector(None, None, (8, 6), square_size),
            VTKGenerator(1920, 1080, [np.array([0, 0, 0])], [Rotation.from_rotvec([0, 0, 0])],
                         [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/chessboard.png'], test_camera_matrix,
                         square_size * 11, square_size * 9),
            (0.2, 0.3), 15, 40, 40, Rotation.from_rotvec([180, 0, 0], degrees=True)
        )
    else:
        square_size = 0.1 / 11
        perform_calibration(
            profile,
            ChessboardDetector(None, None, (8, 6), square_size),
            create_manipulator_generator(base2camera_translation, base2camera_rotation),
            (0.2, 0.3), 15, 40, 40, Rotation.from_rotvec([180, 0, 0], degrees=True)
        )

    info = read_profile_json(profile)
    print(f"Got cameraMatrix: {info.get("cameraMatrix")}")
    print(f"Got distortionCoefficients: {info.get("distortionCoefficients")}")

def hand_to_eye_calibration(profile: str, is_virtual: bool, base2camera_translation: np.array = None, base2camera_rotation: Rotation = None):
    calibration_image_settings = create_image_generation_settings('aruco', '')
    used_transform = create_parser(calibration_image_settings, 'single')
    if is_virtual:
        used_generator = create_vtk_generator(calibration_image_settings, used_transform, 'aruco', 'single')
    else:
        used_generator = create_manipulator_generator(base2camera_translation, base2camera_rotation)
    used_detector = create_aruco_detector(profile, calibration_image_settings, False)
    perform_eye_hand(profile, used_detector, used_transform, used_generator, (1.15, 1.3), 5, 5, 30, Rotation.from_rotvec([180, 0, 0], degrees=True), base2camera_translation, base2camera_rotation)

    info = read_profile_json(profile)
    print(f"Got cameraTranslation: {info.get("cameraTranslation")}")
    print(f"Got cameraRotation: {info.get("cameraRotation")}")

def hand_to_eye_calibration_on_profiles(save_to_profile: str, profiles: list[str]):
    translations_from_camera = []
    rotations_from_camera = []
    translations_from_base = []
    rotations_from_base = []
    detected_mask = []

    info = read_info(profiles)
    for profile in profiles:
        names = pd.read_csv(f"{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{image_info_filename}.csv")["imageName"]
        names = [int(name.split('.')[0]) for name in names]
        profile_info = get_info_part(info, profile, {})
        for translation in profile_info["detectedT"]:
            translations_from_camera.append(translation)
        for rotation in profile_info["detectedR"]:
            rotations_from_camera.append(rotation)
        for success in profile_info["isSuccess"]:
            detected_mask.append(success)
        transforms_type = [tr for tr in ["x_y", "x_z", "x_rx", "x_ry", "x_rz", "traj_1", "traj_2"] if tr in profile]
        t, r, _ = create_transforms(np.array([0, 0, 0]), Rotation.from_rotvec([0, 0, 0]), transforms_type[0])
        for index in names:
            translations_from_base.append(t[index])
            rotations_from_base.append(r[index].as_rotvec(degrees=False))

    cameraTranslation, cameraRotation = _run_calibration(
        translations_from_camera,
        rotations_from_camera,
        translations_from_base,
        rotations_from_base,
        detected_mask
    )
    write_info_to_profile_json(save_to_profile, {"cameraTranslation": cameraTranslation, "cameraRotation": cameraRotation})

# if there are images in analyze present than it will have a bug of starting from n.png instead of 0.png
def run_image_info_creation(calibration_profile: str):
    for detector_type in ["aruco", "apriltag"]:
        for setup_type in ["single", "cube"]:
            for transforms_type in ["x_y", "x_z", "x_rx", "x_rz", "traj_1", "traj_2"]:
                profile_str = f"real_{setup_type}_{detector_type}_{transforms_type}"
                profile_folder = f"{os.path.dirname(__file__)}/{generated_info_folder}/{profile_str}"
                try:
                    os.remove(f"{profile_folder}/{image_info_filename}.csv")
                except FileNotFoundError:
                    pass
                try:
                    os.remove(f"{profile_folder}/{detection_info_filename}.csv")
                except FileNotFoundError:
                    pass

                copy_camera_profile_info(calibration_profile, profile_str)
                info = read_profile_json(profile_str)
                image_settings = create_image_generation_settings(detector_type, transforms_type)
                image_settings.clear_existing_images = False
                used_parser = create_parser(image_settings, setup_type)
                used_generator = PseudoImageGenerator(glob.glob(f"{profile_folder}/{analyse_images_folder}/*.png"))
                t, r, s = create_transforms(np.array(info.get("cameraTranslation")),
                                            Rotation.from_rotvec(info.get("cameraRotation"), degrees=False),
                                            transforms_type)
                generate_images(profile_str, used_generator, image_settings, t, r, s)
                create_detections(profile_str, image_settings, used_parser, detector_type, setup_type, transforms_type)

def run_default_calibration(profile: str):
    camera_calibration(profile, False)
    hand_to_eye_calibration(profile, False)
