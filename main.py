import os

import cv2
from scipy.spatial.transform import Rotation

from python.A_calibration import test_run as calibrationTest, perform_calibration
from python.B_handEyeCalibration import test_run as handEyeTest, perform_eye_hand
from python.C_imagesGeneration import test_run as imagesGenerationTest, generate_images, ImageGenerationSettings, \
    test_manipulator
from python.D_tagsDetection import test_run as tagsDetectionTest, perform_detection
from python.E_visualization import simple_show, two_parameter_relation_show, show_missed_count, show_trajectory
from python.models.detectors.apriltagDetector import ApriltagDetector, ApriltagSettings
from python.models.detectors.arucoDetector import ArucoDetector
from python.models.detectors.chessboardDetector import ChessboardDetector
from python.models.imageGenerators.vtkGenerator import VTKGenerator
import numpy as np

from python.models.transformsParser.cubeParser import CubeParser
from python.models.transformsParser.kalmanParser import SimpleKalmanFilterParser
from python.models.transformsParser.physicsParser import SimpleAccelerationConstraintsParser
from python.models.transformsParser.transformsParser import TransformsParser
from python.settings import tag_images_folder, test_camera_matrix
from python.utils import read_profile_json, generate_random_norm_vector, copy_camera_profile_info

def run_default_calibration(profile: str):
    calibration_profile = profile

    square_size = 0.1 / 11
    perform_calibration(
        calibration_profile,
        ChessboardDetector(None, None, (8, 6), square_size),
        VTKGenerator(1920, 1080, [np.array([0, 0, 0])], [Rotation.from_rotvec([0, 0, 0])],
                     [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/chessboard.png'], test_camera_matrix,
                     square_size * 11, square_size * 9),
        (0.2, 0.3), 15, 40, 40, Rotation.from_rotvec([180, 0, 0], degrees=True)
    )

    info = read_profile_json(calibration_profile)
    print(f"Got cameraMatrix: {info.get("cameraMatrix")}")
    print(f"Got distortionCoefficients: {info.get("distortionCoefficients")}")

    calibration_image_settings = ImageGenerationSettings(True, 0.1, True, str(cv2.aruco.DICT_5X5_50), False, "", False)
    parameters = cv2.aruco.DetectorParameters()
    parameters.useAruco3Detection = True
    used_transform = TransformsParser([[0, 0, 0]], [Rotation.from_rotvec([0, 0, 0])], [0])
    used_generator = VTKGenerator(1920, 1080, used_transform.translations, used_transform.rotations,
                                  [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_0.png'],
                                  test_camera_matrix, calibration_image_settings.tagSize * 450 / 354,
                                  calibration_image_settings.tagSize * 450 / 354)
    used_detector = ArucoDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")),
                                  calibration_image_settings.tagSize, parameters, cv2.aruco.DICT_5X5_50)
    perform_eye_hand(calibration_profile, used_detector, used_transform, used_generator, (0.6, 0.8), 18, 40, 30,
                     Rotation.from_rotvec([180, 0, 0], degrees=True))
    info = read_profile_json(calibration_profile)
    print(f"Got cameraTranslation: {info.get("cameraTranslation")}")
    print(f"Got cameraRotation: {info.get("cameraRotation")}")

def test_aruco_simple():
    profile_to_use = "test"
    run_default_calibration(profile_to_use)

    info = read_profile_json(profile_to_use)

    image_settings = ImageGenerationSettings(True, 0.1, True, str(cv2.aruco.DICT_5X5_50), False, "")
    used_detector = ArucoDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")), 0.1,
                                  cv2.aruco.DetectorParameters(), cv2.aruco.DICT_5X5_50)
    used_transform = TransformsParser([[0, 0, 0]], [Rotation.from_rotvec([0, 0, 0])], [0])
    used_generator = VTKGenerator(1920, 1080, used_transform.translations, used_transform.rotations,
                                  [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_0.png'],
                                  test_camera_matrix, image_settings.tagSize * 450 / 354,
                                  image_settings.tagSize * 450 / 354)

    entries_to_make = 10
    translations = [np.array([0, 0, 1]) + generate_random_norm_vector() * 0.1 for i in range(entries_to_make)]
    rotations = [
        Rotation.from_rotvec(generate_random_norm_vector() * 80, degrees=True) * Rotation.from_rotvec([180, 0, 0],
                                                                                                      degrees=True) for
        i in range(entries_to_make)]
    generate_images(profile_to_use, used_generator, image_settings, translations, rotations)

    perform_detection(profile_to_use, used_detector, used_transform, True)

    simple_show([profile_to_use])

def test_apriltag_simple():
    profile_to_use = "test"
    run_default_calibration(profile_to_use)

    info = read_profile_json(profile_to_use)

    image_settings = ImageGenerationSettings(True, 0.1, False, "", True, "tag36h11")
    used_detector = ApriltagDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")),
                                     0.1, ApriltagSettings(), image_settings.apriltagFamily)
    used_transform = TransformsParser([[0, 0, 0]], [Rotation.from_rotvec([0, 0, 0])], [0])
    used_generator = VTKGenerator(1920, 1080, used_transform.translations, used_transform.rotations,
                                  [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/april_36h11_0.png'],
                                  test_camera_matrix, image_settings.tagSize * 450 / 354,
                                  image_settings.tagSize * 450 / 354)

    entries_to_make = 10
    translations = [np.array([0, 0, 1]) + generate_random_norm_vector() * 0.1 for i in range(entries_to_make)]
    rotations = [
        Rotation.from_rotvec(generate_random_norm_vector() * 30, degrees=True) * Rotation.from_rotvec([180, 0, 0],
                                                                                                      degrees=True) for
        i in range(entries_to_make)]
    generate_images(profile_to_use, used_generator, image_settings, translations, rotations)

    perform_detection(profile_to_use, used_detector, used_transform, True)

    simple_show([profile_to_use])

def test_aruco_cube():
    run_default_calibration("test")
    profile_to_use = "test"

    info = read_profile_json(profile_to_use)

    image_settings = ImageGenerationSettings(True, 0.1, True, str(cv2.aruco.DICT_5X5_50), False, "")
    used_detector = ArucoDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")), 0.1,
                                  cv2.aruco.DetectorParameters(), cv2.aruco.DICT_5X5_50)
    used_transform = CubeParser([0, 1, 2, 3, 4, 5], image_settings.tagSize * 450 / 354)
    used_generator = VTKGenerator(1920, 1080, used_transform.translations, used_transform.rotations,
                                  [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_0.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_1.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_2.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_3.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_4.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_5.png'],
                                  test_camera_matrix, image_settings.tagSize * 450 / 354,
                                  image_settings.tagSize * 450 / 354)

    entries_to_make = 10
    translations = [np.array([0, 0, 1]) + generate_random_norm_vector() * 0.1 for i in range(entries_to_make)]
    rotations = [
        Rotation.from_rotvec(generate_random_norm_vector() * 80, degrees=True) * Rotation.from_rotvec([180, 0, 0],
                                                                                                      degrees=True) for
        i in range(entries_to_make)]
    generate_images(profile_to_use, used_generator, image_settings, translations, rotations)

    perform_detection(profile_to_use, used_detector, used_transform, True)

    simple_show([profile_to_use])

def test_apriltag_cube():
    profile_to_use = "test"
    run_default_calibration(profile_to_use)

    info = read_profile_json(profile_to_use)

    image_settings = ImageGenerationSettings(True, 0.1, False, "", True, "tag36h11")
    used_detector = ApriltagDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")),
                                     0.1, ApriltagSettings(), image_settings.apriltagFamily)
    used_transform = CubeParser([0, 1, 2, 3, 4, 5], image_settings.tagSize * 450 / 354)
    used_generator = VTKGenerator(1920, 1080, used_transform.translations, used_transform.rotations,
                                  [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/april_36h11_0.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/april_36h11_1.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/april_36h11_2.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/april_36h11_3.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/april_36h11_4.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/april_36h11_5.png'],
                                  test_camera_matrix, image_settings.tagSize * 450 / 354,
                                  image_settings.tagSize * 450 / 354)

    entries_to_make = 10
    translations = [np.array([0, 0, 1]) + generate_random_norm_vector() * 0.1 for i in range(entries_to_make)]
    rotations = [
        Rotation.from_rotvec(generate_random_norm_vector() * 80, degrees=True) * Rotation.from_rotvec([180, 0, 0],
                                                                                                      degrees=True) for
        i in range(entries_to_make)]
    generate_images(profile_to_use, used_generator, image_settings, translations, rotations)

    perform_detection(profile_to_use, used_detector, used_transform, True)

    simple_show([profile_to_use])

def x_y_experiment(distance: float, deviation: float, entries_per_axis: int) -> (list[list[float]], list[Rotation]):
    translations = []
    rotations = []

    for x in np.linspace(-deviation, deviation, entries_per_axis):
        for y in np.linspace(-deviation, deviation, entries_per_axis):
            translations.append([x, y, distance])
            rotations.append(Rotation.from_rotvec([0, 0, 0], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
    return translations, rotations, None

def x_z_experiment(distance: float, deviation: float, entries_per_axis: int) -> (list[list[float]], list[Rotation]):
    translations = []
    rotations = []

    for x in np.linspace(-deviation, deviation, entries_per_axis):
        for z in np.linspace(-deviation, deviation, entries_per_axis):
            translations.append([x, 0, distance + z])
            rotations.append(Rotation.from_rotvec([0, 0, 0], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
    return translations, rotations, None

def x_rx_experiment(distance: float, deviation: float, angle_deviation: float, entries_per_translation: int, entries_per_rotation: int) -> (list[list[float]], list[Rotation]):
    translations = []
    rotations = []

    for x in np.linspace(-deviation, deviation, entries_per_translation):
        for rx in np.linspace(-angle_deviation, angle_deviation, entries_per_rotation):
            translations.append([x, 0, distance])
            rotations.append(Rotation.from_rotvec([rx, 0, 0], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
    return translations, rotations, None

def x_ry_experiment(distance: float, deviation: float, angle_deviation: float, entries_per_translation: int, entries_per_rotation: int) -> (list[list[float]], list[Rotation]):
    translations = []
    rotations = []

    for x in np.linspace(-deviation, deviation, entries_per_translation):
        for ry in np.linspace(-angle_deviation, angle_deviation, entries_per_rotation):
            translations.append([x, 0, distance])
            rotations.append(Rotation.from_rotvec([0, ry, 0], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
    return translations, rotations, None

def x_rz_experiment(distance: float, deviation: float, angle_deviation: float, entries_per_translation: int, entries_per_rotation: int) -> (list[list[float]], list[Rotation]):
    translations = []
    rotations = []

    for x in np.linspace(-deviation, deviation, entries_per_translation):
        for rz in np.linspace(-angle_deviation, angle_deviation, entries_per_rotation):
            translations.append([x, 0, distance])
            rotations.append(Rotation.from_rotvec([0, 0, rz], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
    return translations, rotations, None

def simple_trajectory_experiment(z: float) -> (list[list[float]], list[Rotation], list[float]):
    translations = []
    rotations = []

    pos = -0.2
    speed = 0
    acceleration = 0.05
    frame_time = 1/20
    for t in range(0, int(3.9/frame_time)):
        speed += acceleration * frame_time * 0.5
        translations.append([pos + speed * frame_time, 0, z])
        speed += acceleration * frame_time * 0.5
        pos = translations[-1][0]
        rotations.append(Rotation.from_rotvec([0, 0, 0], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
    acceleration = -1
    for t in range(0, int(0.2/frame_time)):
        speed += acceleration * frame_time * 0.5
        translations.append([pos + speed * frame_time, 0, z])
        speed += acceleration * frame_time * 0.5
        pos = translations[-1][0]
        rotations.append(Rotation.from_rotvec([0, 0, 0], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
    acceleration = -0.05
    for t in range(0, int(3.8 / frame_time) + 1):
        speed += acceleration * frame_time * 0.5
        translations.append([pos + speed * frame_time, 0, z])
        speed += acceleration * frame_time * 0.5
        pos = translations[-1][0]
        rotations.append(
            Rotation.from_rotvec([0, 0, 0], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
    acceleration = 1
    for t in range(0, int(0.2 / frame_time)):
        speed += acceleration * frame_time * 0.5
        translations.append([pos + speed * frame_time, 0, z])
        speed += acceleration * frame_time * 0.5
        pos = translations[-1][0]
        rotations.append(Rotation.from_rotvec([0, 0, 0], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
    return translations, rotations, (np.arange(0, len(translations)) * frame_time).tolist()

def simple_trajectory_rotation_experiment(z: float) -> (list[list[float]], list[Rotation], list[float]):
    translations = []
    rotations = []

    pos = -0.2
    speed = 0
    acceleration = 0.05
    frame_time = 1/20
    total_time = 0
    for t in range(0, int(3/frame_time)):
        speed += acceleration * frame_time * 0.5
        translations.append([pos + speed * frame_time, np.sin(total_time * 0.9) * 0.05, z])
        speed += acceleration * frame_time * 0.5
        pos = translations[-1][0]
        rotations.append(Rotation.from_rotvec([np.sin(total_time * 2) * 70, np.sin(total_time * 2.6 - 0.5) * 70, 0], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
        total_time += frame_time
    acceleration = -0.05
    for t in range(0, int(3.9/frame_time)):
        speed += acceleration * frame_time * 0.5
        translations.append([pos + speed * frame_time, np.sin(total_time * 0.9) * 0.05, z])
        speed += acceleration * frame_time * 0.5
        pos = translations[-1][0]
        rotations.append(Rotation.from_rotvec([np.sin(total_time * 2) * 70, np.sin(total_time * 2.6 - 0.5) * 70, 0], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
        total_time += frame_time
    return translations, rotations, (np.arange(0, len(translations)) * frame_time).tolist()

def simple_trajectory_only_rotate_experiment(z: float) -> (list[list[float]], list[Rotation], list[float]):
    translations = []
    rotations = []

    frame_time = 1/20
    total_time = 0
    for t in range(0, int(3/frame_time)):
        translations.append([0, 0, z])
        rotations.append(Rotation.from_rotvec([((-1.5 + total_time) / 1.5) * 90, 0, 0], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
        total_time += frame_time
    return translations, rotations, (np.arange(0, len(translations)) * frame_time).tolist()

def experiments_test():
    image_settings = ImageGenerationSettings(True, 0.1, True, str(cv2.aruco.DICT_5X5_50), False, "", False)
    profiles_to_use = ["x_y", "x_z", "x_rx", "x_ry", "x_rz", "traj_1", "traj_2", "traj_3"]
    profiles_transforms = [
        x_y_experiment(0.8, 0.2, 15),
        x_z_experiment(0.8, 0.2, 15),
        x_rx_experiment(0.8, 0.2, 50, 15, 10),
        x_ry_experiment(0.8, 0.2, 50, 15, 10),
        x_rz_experiment(0.8, 0.2, 50, 15, 10),
        simple_trajectory_experiment(0.8),
        simple_trajectory_rotation_experiment(0.8),
        simple_trajectory_only_rotate_experiment(2.5)
    ]

    info = read_profile_json(profiles_to_use[0])

    parameters = cv2.aruco.DetectorParameters()
    parameters.useAruco3Detection = True
    used_detector = ArucoDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")), image_settings.tagSize, parameters, cv2.aruco.DICT_5X5_50)
    used_transform = CubeParser([0, 1, 2, 3, 4, 5], image_settings.tagSize * 450 / 354)
    used_generator = VTKGenerator(1920, 1080, used_transform.translations, used_transform.rotations,
                                  [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_0.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_1.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_2.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_3.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_4.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_5.png'],
                                  test_camera_matrix, image_settings.tagSize * 450 / 354,
                                  image_settings.tagSize * 450 / 354)

    '''
    ALTERNATIVE SETUP
    used_transform = TransformsParser([[0, 0, 0]], [Rotation.from_rotvec([0, 0, 0])], [0])
    used_generator = VTKGenerator(1920, 1080, used_transform.translations, used_transform.rotations,
                                  [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_0.png'],
                                  test_camera_matrix, image_settings.tagSize * 450 / 354,
                                  image_settings.tagSize * 450 / 354)
    '''

    # for profile in profiles_to_use[1:]:
    #     copy_camera_profile_info(profiles_to_use[0], profile)

    for i in range(len(profiles_to_use) - 3):
        # generate_images(profiles_to_use[i], used_generator, image_settings, profiles_transforms[i][0], profiles_transforms[i][1])
        perform_detection(profiles_to_use[i], used_detector, used_transform, True)
    image_settings = ImageGenerationSettings(True, 0.1, True, str(cv2.aruco.DICT_5X5_50), False, "", True)
    for i in range(len(profiles_to_use) - 3, len(profiles_to_use)):
        # generate_images(profiles_to_use[i], used_generator, image_settings, profiles_transforms[i][0], profiles_transforms[i][1], profiles_transforms[i][2])
        perform_detection(profiles_to_use[i], used_detector, used_transform, True)

    two_parameter_relation_show(profiles_to_use[0], True, 'x', True, 'y', 0, '_aruco3_cuber', {"aruco3": True})
    two_parameter_relation_show(profiles_to_use[1], True, 'x', True, 'z', 0, '_aruco3_cuber', {"aruco3": True})
    two_parameter_relation_show(profiles_to_use[2], True, 'x', False, 'x', 0, '_aruco3_cuber', {"aruco3": True})
    two_parameter_relation_show(profiles_to_use[3], True, 'x', False, 'y', 0, '_aruco3_cuber', {"aruco3": True})
    two_parameter_relation_show(profiles_to_use[4], True, 'x', False, 'z', 0, '_aruco3_cuber', {"aruco3": True})
    two_parameter_relation_show(profiles_to_use[5], True, 'x', True, 'x', 0, '_aruco3_cuber', {"aruco3": True})
    two_parameter_relation_show(profiles_to_use[6], True, 'x', True, 'x', 0, '_aruco3_cuber', {"aruco3": True})
    two_parameter_relation_show(profiles_to_use[7], False, 'x', False, 'x', 180, '_aruco3_cuber', {"aruco3": True})

def physics_parser_test():
    image_settings = ImageGenerationSettings(True, 0.1, True, str(cv2.aruco.DICT_5X5_50), False, "", True)
    profiles_to_use = ["traj_1", "traj_2", "traj_3"]

    info = read_profile_json(profiles_to_use[0])
    parameters = cv2.aruco.DetectorParameters()
    parameters.useAruco3Detection = True
    used_detector = ArucoDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")),
                                  image_settings.tagSize, parameters, cv2.aruco.DICT_5X5_50)
    used_transform = CubeParser([0, 1, 2, 3, 4, 5], image_settings.tagSize * 450 / 354)
    # used_transform = TransformsParser([[0, 0, 0]], [Rotation.from_rotvec([0, 0, 0])], [0])

    for i in range(0, len(profiles_to_use)):
        physics_transform = SimpleKalmanFilterParser(
            used_transform,
            1,
            False,
            True
        )
        perform_detection(profiles_to_use[i], used_detector, physics_transform, True)

    two_parameter_relation_show(profiles_to_use[0], True, 't', True, 'x', 0, '_aruco3_phys_cube_filter', {"aruco3": True})
    show_trajectory(profiles_to_use[0], True, 'x', '_aruco3_phys_cube_filter', {"aruco3": True})
    two_parameter_relation_show(profiles_to_use[1], True, 't', True, 'x', 0, '_aruco3_phys_cube_filter', {"aruco3": True})
    two_parameter_relation_show(profiles_to_use[2], False, 'x', False, 'x', 180, '_aruco3_phys_cube_filter', {"aruco3": True})
    show_trajectory(profiles_to_use[2], False, 'x', '_aruco3_phys_cube_filter', {"aruco3": True})

def generate_virtual_images():
    def create_image_generation_settings(detector_type: str, transforms_type: str) -> ImageGenerationSettings:
        return ImageGenerationSettings(
            True,
            0.1,
            detector_type == 'aruco',
            str(cv2.aruco.DICT_5X5_50) if detector_type == 'aruco' else '',
            detector_type == 'apriltag',
            "tag36h11" if detector_type == 'apriltag' else '',
            "traj" in transforms_type
        )

    def create_parser(settings: ImageGenerationSettings, setup_type: str) -> TransformsParser:
        if setup_type == "single":
            return TransformsParser([[0, 0, 0]], [Rotation.from_rotvec([0, 0, 0])], [0])
        else:
            return CubeParser([0, 1, 2, 3, 4, 5], settings.tagSize * 450 / 354)

    def create_generator(settings: ImageGenerationSettings, used_transform: TransformsParser, detector_type: str, setup_type: str) -> VTKGenerator:
        if detector_type == "aruco":
            path_creator = lambda i: f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_{i}.png'
        else:
            path_creator = lambda i: f'{os.path.dirname(__file__)}/python/{tag_images_folder}/april_36h11_{i}.png'
        paths = [path_creator(i) for i in range(0, 1 if setup_type == "single" else 6)]
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

    def create_transforms(transforms_type: str, distance: float):
        if transforms_type == "x_y":
            return x_y_experiment(distance, 0.2, 15)
        elif transforms_type == "x_z":
            return x_z_experiment(distance, 0.2, 15)
        elif transforms_type == "x_rx":
            return x_rx_experiment(distance, 0.2, 50, 15, 10)
        elif transforms_type == "x_ry":
            return x_ry_experiment(distance, 0.2, 50, 15, 10)
        elif transforms_type == "x_rz":
            return x_rz_experiment(distance, 0.2, 50, 15, 10)
        elif transforms_type == "traj_1":
            return simple_trajectory_experiment(distance)
        elif transforms_type == "traj_2":
            return simple_trajectory_rotation_experiment(distance)

    def create_detections(profile: str, settings: ImageGenerationSettings, parser: TransformsParser, detector_type: str, setup_type: str, transforms_type: str):
        info = read_profile_json(profile)
        used_detectors = []
        if detector_type == "aruco":
            for aruco3 in [False, True]:
                parameters = cv2.aruco.DetectorParameters()
                parameters.useAruco3Detection = aruco3
                used_detectors.append(ArucoDetector(
                    np.array(info.get("cameraMatrix")),
                    np.array(info.get("distortionCoefficients")),
                    settings.tagSize,
                    parameters,
                    cv2.aruco.DICT_5X5_50
                ))
        else:
            used_detectors.append(ApriltagDetector(
                np.array(info.get("cameraMatrix")),
                np.array(info.get("distortionCoefficients")),
                settings.tagSize,
                ApriltagSettings(),
                settings.apriltagFamily
            ))

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


    calibration_profile = "calibration"
    run_default_calibration(calibration_profile)

    for detector_type in ["aruco", "apriltag"]:
        for setup_type in ["single", "cube"]:
            for transforms_type in ["x_y", "x_z", "x_rx", "x_ry", "x_rz", "traj_1", "traj_2"]:
                for distance in [0.8, 2.0]:
                    profile_str = f"{setup_type}_{detector_type}_{transforms_type}_{"close" if distance < 1 else "far"}"
                    copy_camera_profile_info(calibration_profile, profile_str)
                    image_settings = create_image_generation_settings(detector_type, transforms_type)
                    used_parser = create_parser(image_settings, setup_type)
                    used_generator = create_generator(image_settings, used_parser, detector_type, setup_type)
                    t, r, s = create_transforms(transforms_type, distance)
                    generate_images(profile_str, used_generator, image_settings, t, r, s)
                    create_detections(profile_str, image_settings, used_parser, detector_type, setup_type, transforms_type)


if __name__ == "__main__":
    # calibrationTest()
    # handEyeTest()
    # imagesGenerationTest()
    # tagsDetectionTest()

    #test_aruco_cube()

    #test_manipulator('192.168.1.101', 30002)

    # experiments_test()
    # physics_parser_test()
    generate_virtual_images()

