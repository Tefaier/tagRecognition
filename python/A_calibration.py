import os
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
from python.settings import generated_info_folder, calibration_images_folder, image_width, image_height, tag_images_folder, \
    test_camera_matrix, general_info_filename
from python.utils import ensure_folder_exists, get_gray_image, generate_random_norm_vector, update_json

def prepare_folder(path: str):
    ensure_folder_exists(path)
    files = glob.glob(f"{path}/*")
    for f in files:
        os.remove(f)

def perform_calibration(profile: str, detector: TagDetector, generator: ImageGenerator) -> (list, list):
    prepare_folder(f"{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{calibration_images_folder}")

    # position around which images are created
    index = 0
    base_translation = np.array([0, 0, 0.15])
    base_rotation = Rotation.from_rotvec([180, 0, 0], degrees=True)
    position_samples = 5
    rotation_samples = 10
    deviation_translation = np.array([0.08, 0.02, 0.05])  # in meters
    angle_rotation = 50  # in degrees
    for _ in range(0, position_samples):
        translation = base_translation + generate_random_norm_vector() * deviation_translation
        for _ in range(0, rotation_samples):
            rotation = base_rotation * Rotation.from_rotvec(generate_random_norm_vector() * angle_rotation, degrees=True)
            generator.generate_image_with_obj_at_transform(translation, rotation, f'{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{calibration_images_folder}/{index}.png')
            index += 1

    cameraMatrix, distortionCoefficients = perform_calibration_on_existing_images(profile, detector)
    return cameraMatrix, distortionCoefficients

def perform_calibration_on_existing_images(profile: str, detector: TagDetector) -> (list, list):
    ensure_folder_exists(f"{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{calibration_images_folder}")
    images = glob.glob(f'{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{calibration_images_folder}/*.png')

    objpoints = []
    imgpoints = []
    for name in images:
        image = cv2.imread(name)
        objp, imgp = detector.detect_object_points(image)
        if imgp is not None:
            objpoints.append(objp)
            imgpoints.append(imgp)

    ret, cameraMatrix, distortionCoefficients, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, get_gray_image(cv2.imread(images[-1])).shape[::-1], None, None, flags = cv2.CALIB_USE_LU)
    cameraMatrix = cameraMatrix.reshape((3, 3)).tolist()
    distortionCoefficients = distortionCoefficients.reshape((5,)).tolist()
    update_json({"cameraMatrix": cameraMatrix, "distortionCoefficients": distortionCoefficients},
               f'{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{general_info_filename}.json')
    return cameraMatrix, distortionCoefficients

def test_run():
    chessboard_pattern = (8, 6)
    pattern_width = 0.1
    pattern_height = 0.1 * 9 / 11
    square_size = pattern_width / 11
    # performCalibrationOnExistingImages("test", pattern_width, ChessboardDetector(None, None, chessboard_pattern, square_size))
    perform_calibration(
        "test",
        ChessboardDetector(None, None, chessboard_pattern, square_size),
        VTKGenerator(image_width, image_height, f'{os.path.dirname(__file__)}/{tag_images_folder}/chessboard.png', test_camera_matrix, pattern_width,
                     pattern_height)
    )
    # performCalibration(
    #     "test",
    #     ArucoDetector(None, None, pattern_width, cv2.aruco.DICT_5X5_50),
    #     VTKGenerator(imageWidth, imageHeight, f'{os.path.dirname(__file__)}/{tagImagesFolder}/aruco_1.png', testCameraMatrix, pattern_width,
    #                  pattern_width)
    # )
