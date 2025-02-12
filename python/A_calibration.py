import os
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
from python.settings import generated_info_folder, calibration_images_folder, image_width, image_height, tag_images_folder, \
    test_camera_matrix
from python.utils import ensure_folder_exists, get_gray_image, generate_random_norm_vector, write_info_to_profile_json, \
    random_generator


def prepare_folder(path: str):
    ensure_folder_exists(path)
    files = glob.glob(f"{path}/*")
    for f in files:
        os.remove(f)

# angles are in degrees
def perform_calibration(profile: str, detector: TagDetector, generator: ImageGenerator, distance_range: Tuple[float, float], x_deviation_angle: float, y_deviation_angle: float, obj_rotation_limit: float, rotate_from: Rotation) -> (list, list):
    prepare_folder(f"{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{calibration_images_folder}")

    # position around which images are created
    index = 0
    position_samples = 5
    rotation_samples = 10
    for _ in range(0, position_samples):
        translation = Rotation.from_euler('xyz', [random_generator.uniform(-x_deviation_angle, x_deviation_angle), random_generator.uniform(-y_deviation_angle, y_deviation_angle), 0], degrees=True).apply(np.array([0, 0, random_generator.uniform(distance_range[0], distance_range[1])]))
        for _ in range(0, rotation_samples):
            rotation = Rotation.from_rotvec(generate_random_norm_vector() * obj_rotation_limit, degrees=True) * rotate_from
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
    write_info_to_profile_json(profile, {"cameraMatrix": cameraMatrix, "distortionCoefficients": distortionCoefficients})
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
        VTKGenerator(image_width, image_height, [np.array([0, 0, 0])], [Rotation.from_rotvec([0, 0, 0])], [f'{os.path.dirname(__file__)}/{tag_images_folder}/chessboard.png'], test_camera_matrix, pattern_width,
                     pattern_height), (0.2, 0.3), 20, 10, 60, Rotation.from_rotvec([180, 0, 0], degrees=True)
    )
    # performCalibration(
    #     "test",
    #     ArucoDetector(None, None, pattern_width, cv2.aruco.DICT_5X5_50),
    #     VTKGenerator(imageWidth, imageHeight, f'{os.path.dirname(__file__)}/{tagImagesFolder}/aruco_1.png', testCameraMatrix, pattern_width,
    #                  pattern_width)
    # )
