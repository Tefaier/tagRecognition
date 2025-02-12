import math
import os
from math import degrees

import cv2
from scipy.spatial.transform import Rotation

from python.A_calibration import test_run as calibrationTest, perform_calibration
from python.B_handEyeCalibration import test_run as handEyeTest, perform_eye_hand
from python.C_imagesGeneration import test_run as imagesGenerationTest, generate_images, ImageGenerationSettings
from python.D_tagsDetection import test_run as tagsDetectionTest, perform_detection
from python.E_visualization import simple_show
from python.models.detectors.apriltagDetector import ApriltagDetector, ApriltagSettings
from python.models.detectors.arucoDetector import ArucoDetector
from python.models.detectors.chessboardDetector import ChessboardDetector
from python.models.imageGenerators.vtkGenerator import VTKGenerator
import numpy as np

from python.models.transformsParser.cubeParser import CubeParser
from python.models.transformsParser.transformsParser import TransformsParser
from python.settings import tag_images_folder, test_camera_matrix
from python.utils import read_profile_json, generate_random_norm_vector

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
        (0.2, 0.3), 20, 10, 60, Rotation.from_rotvec([180, 0, 0], degrees=True)
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
    generate_images(profile_to_use, used_generator, image_settings, translations, rotations, 1)

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
    generate_images(profile_to_use, used_generator, image_settings, translations, rotations, 1)

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
        (0.2, 0.3), 20, 10, 60, Rotation.from_rotvec([180, 0, 0], degrees=True)
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
    generate_images(profile_to_use, used_generator, image_settings, translations, rotations, 1)

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
        Rotation.from_rotvec(generate_random_norm_vector() * 80, degrees=True) * Rotation.from_rotvec([180, 0, 0],
                                                                                                      degrees=True) for
        i in range(entries_to_make)]
    generate_images(profile_to_use, used_generator, image_settings, translations, rotations, 1)

    perform_detection(profile_to_use, used_detector, used_transform, True)

    simple_show([profile_to_use])

def test_parser():
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

    test_apriltag_cube()

