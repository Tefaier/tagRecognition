import os

import cv2
from scipy.spatial.transform import Rotation

from python.C_imagesGeneration import generate_images, ImageGenerationSettings
from python.D_tagsDetection import perform_detection
from python.E_visualization import simple_show
from python.models.detectors.arucoDetector import ArucoDetector
from python.models.detectors.apriltagDetector import ApriltagDetector, ApriltagSettings
from python.models.imageGenerators.vtkGenerator import VTKGenerator
import numpy as np

from python.models.transformsParser.cubeParser import CubeParser
from python.models.transformsParser.transformsParser import TransformsParser
from python.settings import tag_images_folder, test_camera_matrix
from python.utils import read_profile_json, generate_random_norm_vector
from python.workflow_utils import run_default_calibration


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