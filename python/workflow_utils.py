import os

import cv2
from scipy.spatial.transform import Rotation

from python.A_calibration import perform_calibration
from python.B_handEyeCalibration import perform_eye_hand
from python.C_imagesGeneration import ImageGenerationSettings
from python.D_tagsDetection import perform_detection
from python.experiments import x_y_experiment, x_z_experiment, x_rx_experiment, x_ry_experiment, x_rz_experiment, \
    simple_trajectory_experiment, simple_trajectory_rotation_experiment
from python.models.detectors.arucoDetector import ArucoDetector
from python.models.detectors.apriltagDetector import ApriltagDetector, ApriltagSettings
from python.models.detectors.chessboardDetector import ChessboardDetector
from python.models.imageGenerators.manipulatorGenerator import ManipulatorGenerator
from python.models.imageGenerators.vtkGenerator import VTKGenerator
import numpy as np

from python.models.transformsParser.cubeParser import CubeParser
from python.models.transformsParser.kalmanParser import SimpleKalmanFilterParser
from python.models.transformsParser.transformsParser import TransformsParser
from python.settings import tag_images_folder, test_camera_matrix
from python.utils import read_profile_json, change_base2gripper_to_camera2object

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
        return TransformsParser([[0, 0, settings.tagSize * 450 / 354 / 2]], [Rotation.from_rotvec([0, 0, 90])], [2])
    else:
        return CubeParser([0, 1, 2, 3, 4, 5], settings.tagSize * 450 / 354)

# detector_type either aruco or apriltag
# setup_type either single or cube
def create_vtk_generator(
        settings: ImageGenerationSettings,
        used_transform: TransformsParser,
        detector_type: str,
        setup_type: str
) -> VTKGenerator:
    if detector_type == "aruco":
        path_creator = lambda i: f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_{i}.png'
    else:
        path_creator = lambda i: f'{os.path.dirname(__file__)}/python/{tag_images_folder}/april_36h11_{i}.png'
    paths = [2] if setup_type == "single" else [0, 1, 2, 3, 4, 5]
    paths = [path_creator(path) for path in paths]
    return VTKGenerator(
        1920,
        1080,
        used_transform.translations,
        used_transform.rotations,
        paths,
        test_camera_matrix,
        settings.tagSize * 450 / 354,
        settings.tagSize * 450 / 354
    )

def create_manipulator_generator(
        base2camera_translation: np.array,
        base2camera_rotation: Rotation,
) -> ManipulatorGenerator:
    return ManipulatorGenerator(
        '192.168.56.101',
        30003,
        base2camera_translation,
        base2camera_rotation,
        np.array([0, 0, 0.107]),
        Rotation.from_rotvec([0, 0, 0]),
        0,
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

def create_apriltag_detector(profile: str, settings: ImageGenerationSettings) -> ApriltagDetector:
    info = read_profile_json(profile)
    return ApriltagDetector(
        np.array(info.get("cameraMatrix")),
        np.array(info.get("distortionCoefficients")),
        settings.tagSize,
        ApriltagSettings(),
        settings.apriltagFamily
    )

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
        t, r, s = x_rx_experiment(15, 10)
    elif transforms_type == "x_ry":
        t, r, s = x_ry_experiment(15, 10)
    elif transforms_type == "x_rz":
        t, r, s = x_rz_experiment(15, 10)
    elif transforms_type == "traj_1":
        t, r, s = simple_trajectory_experiment(20)
    elif transforms_type == "traj_2":
        t, r, s = simple_trajectory_rotation_experiment(20)
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
    extra_infos = [{"parser": "simple"}]
    if "traj" in transforms_type:
        for flip in [False, True]:
            for filter in [False] if setup_type == "single" else [False, True]:
                used_parsers.append(SimpleKalmanFilterParser(
                    parser,
                    1,
                    flip,
                    filter
                ))
                extra_infos.append({"parser": "Kalman", "flip": flip, "filter": filter})

    iteration = 0
    for detector in used_detectors:
        for i in range(0, len(used_parsers)):
            perform_detection(profile, detector, used_parsers[i], iteration == 0, extra_infos[i])
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
    perform_eye_hand(profile, used_detector, used_transform, used_generator, (0.6, 0.8), 18, 40, 30, Rotation.from_rotvec([180, 0, 0], degrees=True))

    info = read_profile_json(profile)
    print(f"Got cameraTranslation: {info.get("cameraTranslation")}")
    print(f"Got cameraRotation: {info.get("cameraRotation")}")

def run_default_calibration(profile: str):
    camera_calibration(profile, False)
    hand_to_eye_calibration(profile, False)
