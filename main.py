import os

import cv2
from scipy.spatial.transform import Rotation

from python.A_calibration import test_run as calibrationTest, perform_calibration
from python.B_handEyeCalibration import test_run as handEyeTest, perform_eye_hand
from python.C_imagesGeneration import test_run as imagesGenerationTest, generate_images, ImageGenerationSettings, \
    test_manipulator
from python.D_tagsDetection import test_run as tagsDetectionTest, perform_detection
from python.E_visualization import simple_show, two_parameter_relation_show, show_missed_count, show_trajectory
#from python.models.detectors.apriltagDetector import ApriltagDetector, ApriltagSettings
from python.models.detectors.arucoDetector import ArucoDetector
from python.models.detectors.chessboardDetector import ChessboardDetector
from python.models.imageGenerators.vtkGenerator import VTKGenerator
import numpy as np

from python.models.transformsParser.cubeParser import CubeParser
from python.models.transformsParser.physicsParser import SimpleAccelerationConstraintsParser
from python.models.transformsParser.transformsParser import TransformsParser
from python.settings import tag_images_folder, test_camera_matrix
from python.utils import read_profile_json, generate_random_norm_vector, copy_camera_profile_info


def test_aruco_simple():
    image_settings = ImageGenerationSettings(True, 0.1, True, str(cv2.aruco.DICT_5X5_50), False, "")
    profile_to_use = "test"

    square_size = 0.1 / 11
    perform_calibration(
        profile_to_use,
        ChessboardDetector(None, None, (8, 6), square_size),
        VTKGenerator(1920, 1080, [np.array([0, 0, 0])], [Rotation.from_rotvec([0, 0, 0])],
                     [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/chessboard.png'], test_camera_matrix,
                     square_size * 11, square_size * 9),
        (0.2, 0.3), 15, 5, 30, Rotation.from_rotvec([180, 0, 0], degrees=True)
    )

    info = read_profile_json(profile_to_use)
    print(f"Got cameraMatrix: {info.get("cameraMatrix")}")
    print(f"Got distortionCoefficients: {info.get("distortionCoefficients")}")

    used_detector = ArucoDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")), 0.1,
                                  cv2.aruco.DetectorParameters(), cv2.aruco.DICT_5X5_50)
    used_transform = TransformsParser([[0, 0, 0]], [Rotation.from_rotvec([0, 0, 0])], [0])
    used_generator = VTKGenerator(1920, 1080, used_transform.translations, used_transform.rotations,
                                  [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_0.png'],
                                  test_camera_matrix, image_settings.tagSize * 450 / 354,
                                  image_settings.tagSize * 450 / 354)

    perform_eye_hand(profile_to_use, used_detector, used_transform, used_generator, (0.2, 0.3), 20, 10, 60, Rotation.from_rotvec([180, 0, 0], degrees=True))
    info = read_profile_json(profile_to_use)
    print(f"Got cameraTranslation: {info.get("cameraTranslation")}")
    print(f"Got cameraRotation: {info.get("cameraRotation")}")

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
    image_settings = ImageGenerationSettings(True, 0.1, False, "", True, "tag36h11")
    profile_to_use = "test"

    square_size = 0.1 / 11
    perform_calibration(
        profile_to_use,
        ChessboardDetector(None, None, (8, 6), square_size),
        VTKGenerator(1920, 1080, [np.array([0, 0, 0])], [Rotation.from_rotvec([0, 0, 0])],
                     [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/chessboard.png'], test_camera_matrix,
                     square_size * 11, square_size * 9),
        (0.2, 0.3), 20, 10, 60, Rotation.from_rotvec([180, 0, 0], degrees=True)
    )

    info = read_profile_json(profile_to_use)
    print(f"Got cameraMatrix: {info.get("cameraMatrix")}")
    print(f"Got distortionCoefficients: {info.get("distortionCoefficients")}")

    used_detector = ApriltagDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")),
                                     0.1, ApriltagSettings(), image_settings.apriltagFamily)
    used_transform = TransformsParser([[0, 0, 0]], [Rotation.from_rotvec([0, 0, 0])], [0])
    used_generator = VTKGenerator(1920, 1080, used_transform.translations, used_transform.rotations,
                                  [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/april_36h11_0.png'],
                                  test_camera_matrix, image_settings.tagSize * 450 / 354,
                                  image_settings.tagSize * 450 / 354)

    perform_eye_hand(profile_to_use, used_detector, used_transform, used_generator, (0.2, 0.3), 20, 10, 60, Rotation.from_rotvec([180, 0, 0], degrees=True))
    info = read_profile_json(profile_to_use)
    print(f"Got cameraTranslation: {info.get("cameraTranslation")}")
    print(f"Got cameraRotation: {info.get("cameraRotation")}")

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
    image_settings = ImageGenerationSettings(True, 0.1, True, str(cv2.aruco.DICT_5X5_50), False, "")
    profile_to_use = "test"

    square_size = 0.1 / 11
    perform_calibration(
        profile_to_use,
        ChessboardDetector(None, None, (8, 6), square_size),
        VTKGenerator(1920, 1080, [np.array([0, 0, 0])], [Rotation.from_rotvec([0, 0, 0])],
                     [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/chessboard.png'], test_camera_matrix,
                     square_size * 11, square_size * 9),
        (0.2, 0.3), 5, 15, 30, Rotation.from_rotvec([180, 0, 0], degrees=True)
    )

    info = read_profile_json(profile_to_use)
    print(f"Got cameraMatrix: {info.get("cameraMatrix")}")
    print(f"Got distortionCoefficients: {info.get("distortionCoefficients")}")

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

    perform_eye_hand(profile_to_use, used_detector, used_transform, used_generator, (0.2, 0.3), 20, 10, 60, Rotation.from_rotvec([180, 0, 0], degrees=True))
    info = read_profile_json(profile_to_use)
    print(f"Got cameraTranslation: {info.get("cameraTranslation")}")
    print(f"Got cameraRotation: {info.get("cameraRotation")}")

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
    image_settings = ImageGenerationSettings(True, 0.1, False, "", True, "tag36h11")
    profile_to_use = "test"

    square_size = 0.1 / 11
    perform_calibration(
        profile_to_use,
        ChessboardDetector(None, None, (8, 6), square_size),
        VTKGenerator(1920, 1080, [np.array([0, 0, 0])], [Rotation.from_rotvec([0, 0, 0])],
                     [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/chessboard.png'], test_camera_matrix,
                     square_size * 11, square_size * 9),
        (0.2, 0.3), 20, 10, 60, Rotation.from_rotvec([180, 0, 0], degrees=True)
    )

    info = read_profile_json(profile_to_use)
    print(f"Got cameraMatrix: {info.get("cameraMatrix")}")
    print(f"Got distortionCoefficients: {info.get("distortionCoefficients")}")

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

    perform_eye_hand(profile_to_use, used_detector, used_transform, used_generator, (0.2, 0.3), 20, 10, 60, Rotation.from_rotvec([180, 0, 0], degrees=True))
    info = read_profile_json(profile_to_use)
    print(f"Got cameraTranslation: {info.get("cameraTranslation")}")
    print(f"Got cameraRotation: {info.get("cameraRotation")}")

    entries_to_make = 10
    translations = [np.array([0, 0, 1]) + generate_random_norm_vector() * 0.1 for i in range(entries_to_make)]
    rotations = [
        Rotation.from_rotvec(generate_random_norm_vector() * 80, degrees=True) * Rotation.from_rotvec([180, 0, 0],
                                                                                                      degrees=True) for
        i in range(entries_to_make)]
    generate_images(profile_to_use, used_generator, image_settings, translations, rotations)

    perform_detection(profile_to_use, used_detector, used_transform, True)

    simple_show([profile_to_use])

def x_y_experiment(deviation: float, entries_per_axis: int) -> (list[list[float]], list[Rotation]):
    translations = []
    rotations = []

    for x in np.linspace(-deviation, deviation, entries_per_axis):
        for y in np.linspace(-deviation, deviation, entries_per_axis):
            translations.append([x, y, 0.8])
            rotations.append(Rotation.from_rotvec([0, 0, 0], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
    return translations, rotations

def x_z_experiment(deviation: float, entries_per_axis: int) -> (list[list[float]], list[Rotation]):
    translations = []
    rotations = []

    for x in np.linspace(-deviation, deviation, entries_per_axis):
        for z in np.linspace(-deviation, deviation, entries_per_axis):
            translations.append([x, 0, 0.8 + z])
            rotations.append(Rotation.from_rotvec([0, 0, 0], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
    return translations, rotations

def x_rx_experiment(deviation: float, angle_deviation: float, entries_per_translation: int, entries_per_rotation: int) -> (list[list[float]], list[Rotation]):
    translations = []
    rotations = []

    for x in np.linspace(-deviation, deviation, entries_per_translation):
        for rx in np.linspace(-angle_deviation, angle_deviation, entries_per_rotation):
            translations.append([x, 0, 0.8])
            rotations.append(Rotation.from_rotvec([rx, 0, 0], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
    return translations, rotations

def x_ry_experiment(deviation: float, angle_deviation: float, entries_per_translation: int, entries_per_rotation: int) -> (list[list[float]], list[Rotation]):
    translations = []
    rotations = []

    for x in np.linspace(-deviation, deviation, entries_per_translation):
        for ry in np.linspace(-angle_deviation, angle_deviation, entries_per_rotation):
            translations.append([x, 0, 0.8])
            rotations.append(Rotation.from_rotvec([0, ry, 0], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
    return translations, rotations

def x_rz_experiment(deviation: float, angle_deviation: float, entries_per_translation: int, entries_per_rotation: int) -> (list[list[float]], list[Rotation]):
    translations = []
    rotations = []

    for x in np.linspace(-deviation, deviation, entries_per_translation):
        for rz in np.linspace(-angle_deviation, angle_deviation, entries_per_rotation):
            translations.append([x, 0, 0.8])
            rotations.append(Rotation.from_rotvec([0, 0, rz], degrees=True) * Rotation.from_rotvec([180, 0, 0], degrees=True))
    return translations, rotations

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

def experiments_test():
    image_settings = ImageGenerationSettings(True, 0.1, True, str(cv2.aruco.DICT_5X5_50), False, "", False)
    profiles_to_use = ["x_y", "x_z", "x_rx", "x_ry", "x_rz", "traj_1", "traj_2"]
    # profiles_transforms = [
    #     x_y_experiment(0.2, 15),
    #     x_z_experiment(0.2, 15),
    #     x_rx_experiment(0.2, 50, 15, 10),
    #     x_ry_experiment(0.2, 50, 15, 10),
    #     x_rz_experiment(0.2, 50, 15, 10),
    #     simple_trajectory_experiment(0.8),
    #     simple_trajectory_rotation_experiment(0.8)
    # ]
    #
    # square_size = 0.1 / 11
    # perform_calibration(
    #     profiles_to_use[0],
    #     ChessboardDetector(None, None, (8, 6), square_size),
    #     VTKGenerator(1920, 1080, [np.array([0, 0, 0])], [Rotation.from_rotvec([0, 0, 0])],
    #                  [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/chessboard.png'], test_camera_matrix,
    #                  square_size * 11, square_size * 9),
    #     (0.2, 0.3), 15, 40, 40, Rotation.from_rotvec([180, 0, 0], degrees=True)
    # )
    #
    # info = read_profile_json(profiles_to_use[0])
    # print(f"Got cameraMatrix: {info.get("cameraMatrix")}")
    # print(f"Got distortionCoefficients: {info.get("distortionCoefficients")}")
    #
    # used_detector = ArucoDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")), image_settings.tagSize, cv2.aruco.DetectorParameters(), cv2.aruco.DICT_5X5_50)
    # # used_detector = ArucoDetector(test_camera_matrix, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), image_settings.tagSize, cv2.aruco.DetectorParameters(), cv2.aruco.DICT_5X5_50)
    # used_transform = TransformsParser([[0, 0, 0]], [Rotation.from_rotvec([0, 0, 0])], [0])
    # used_generator = VTKGenerator(1920, 1080, used_transform.translations, used_transform.rotations,
    #                               [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_0.png'],
    #                               test_camera_matrix, image_settings.tagSize * 450 / 354,
    #                               image_settings.tagSize * 450 / 354)
    #
    # perform_eye_hand(profiles_to_use[0], used_detector, used_transform, used_generator, (0.6, 0.8), 18, 40, 30, Rotation.from_rotvec([180, 0, 0], degrees=True))
    # info = read_profile_json(profiles_to_use[0])
    # print(f"Got cameraTranslation: {info.get("cameraTranslation")}")
    # print(f"Got cameraRotation: {info.get("cameraRotation")}")
    #
    # for profile in profiles_to_use[1:]:
    #     copy_camera_profile_info(profiles_to_use[0], profile)
    #
    # for i in range(len(profiles_to_use) - 2):
    #     generate_images(profiles_to_use[i], used_generator, image_settings, profiles_transforms[i][0], profiles_transforms[i][1])
    #     perform_detection(profiles_to_use[i], used_detector, used_transform, True)
    # image_settings = ImageGenerationSettings(True, 0.1, True, str(cv2.aruco.DICT_5X5_50), False, "", True)
    # for i in range(len(profiles_to_use) - 2, len(profiles_to_use)):
    #     generate_images(profiles_to_use[i], used_generator, image_settings, profiles_transforms[i][0], profiles_transforms[i][1], profiles_transforms[i][2])
    #     perform_detection(profiles_to_use[i], used_detector, used_transform, True)

    # two_parameter_relation_show(profiles_to_use[0], True, 'x', True, 'y', 0)
    # two_parameter_relation_show(profiles_to_use[1], True, 'x', True, 'z', 0)
    # two_parameter_relation_show(profiles_to_use[2], True, 'x', False, 'x', 0)
    # two_parameter_relation_show(profiles_to_use[3], True, 'x', False, 'y', 0)
    # two_parameter_relation_show(profiles_to_use[4], True, 'x', False, 'z', 0)
    # two_parameter_relation_show(profiles_to_use[5], True, 'x', True, 'x', 0)
    two_parameter_relation_show(profiles_to_use[6], True, 'x', True, 'x', 0)

def physics_parser_test():
    image_settings = ImageGenerationSettings(True, 0.1, True, str(cv2.aruco.DICT_5X5_50), False, "", True)
    profiles_to_use = ["traj_1", "traj_2"]

    info = read_profile_json(profiles_to_use[0])
    used_detector = ArucoDetector(np.array(info.get("cameraMatrix")), np.array(info.get("distortionCoefficients")),
                                  image_settings.tagSize, cv2.aruco.DetectorParameters(), cv2.aruco.DICT_5X5_50)
    used_transform = TransformsParser([[0, 0, 0]], [Rotation.from_rotvec([0, 0, 0])], [0])

    for i in range(0, len(profiles_to_use)):
        physics_transform = SimpleAccelerationConstraintsParser(
            used_transform,
            (-1, 1),
            (-1, 1),
            0.3,
            5,
            0.3,
            True
        )
        perform_detection(profiles_to_use[i], used_detector, physics_transform, True)

    two_parameter_relation_show(profiles_to_use[0], True, 't', True, 'x', 0, '_phys')
    two_parameter_relation_show(profiles_to_use[1], True, 't', True, 'x', 0, '_phys')

def generator_test():
    used_transform = CubeParser([0, 1, 2, 3, 4, 5], 0.1 * 450 / 354)
    used_generator = VTKGenerator(1920, 1080, used_transform.translations, used_transform.rotations,
                                  [f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_0.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_1.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_2.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_3.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_4.png',
                                   f'{os.path.dirname(__file__)}/python/{tag_images_folder}/aruco_5x5_5.png'],
                                  test_camera_matrix, 0.1 * 450 / 354,
                                  0.1 * 450 / 354)
    used_generator.generate_image_with_obj_at_transform(np.array([0, 0, 0.5]), Rotation.from_rotvec([20, -20, 0], degrees=True), "cube.png")

if __name__ == "__main__":
    # calibrationTest()
    # handEyeTest()
    # imagesGenerationTest()
    # tagsDetectionTest()

    #test_aruco_cube()

    #test_manipulator('192.168.1.101', 30002)

    # experiments_test()
    physics_parser_test()

