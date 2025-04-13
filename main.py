import os

import cv2
from scipy.spatial.transform import Rotation

from python.A_calibration import test_run as calibrationTest, perform_calibration
from python.B_handEyeCalibration import test_run as handEyeTest, perform_eye_hand
from python.C_imagesGeneration import test_run as imagesGenerationTest, generate_images, ImageGenerationSettings, \
    test_manipulator
from python.D_tagsDetection import test_run as tagsDetectionTest, perform_detection
from python.E_visualization import simple_show, two_parameter_relation_show, show_missed_count, show_trajectory
from python.experiments import x_y_experiment, x_z_experiment, x_rx_experiment, x_ry_experiment, x_rz_experiment, \
    simple_trajectory_experiment, simple_trajectory_rotation_experiment, simple_trajectory_only_rotate_experiment
# from python.models.detectors.apriltagDetector import ApriltagDetector, ApriltagSettings
from python.models.detectors.arucoDetector import ArucoDetector
from python.models.detectors.chessboardDetector import ChessboardDetector
from python.models.imageGenerators.imageGenerator import ImageGenerator
from python.models.imageGenerators.manipulatorGenerator import ManipulatorGenerator
from python.models.imageGenerators.vtkGenerator import VTKGenerator
import numpy as np

from python.models.transformsParser.cubeParser import CubeParser
from python.models.transformsParser.kalmanParser import SimpleKalmanFilterParser
from python.models.transformsParser.physicsParser import SimpleAccelerationConstraintsParser
from python.models.transformsParser.transformsParser import TransformsParser
from python.settings import tag_images_folder, test_camera_matrix
from python.utils import read_profile_json, generate_random_norm_vector, copy_camera_profile_info, \
    change_base2gripper_to_camera2object
from python.workflow_utils import run_default_calibration, create_image_generation_settings, create_parser, \
    create_vtk_generator, create_transforms, create_detections


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
    calibration_profile = "calibration"
    run_default_calibration(calibration_profile)

    for detector_type in ["aruco", "apriltag"]:
        for setup_type in ["single", "cube"]:
            for transforms_type in ["x_y", "x_z", "x_rx", "x_ry", "x_rz", "traj_1", "traj_2"]:
                for distance in [0.8, 2.0]:
                    profile_str = f"{setup_type}_{detector_type}_{transforms_type}_{"close" if distance < 1 else "far"}"
                    copy_camera_profile_info(calibration_profile, profile_str)
                    info = read_profile_json(profile_str)
                    image_settings = create_image_generation_settings(detector_type, transforms_type)
                    used_parser = create_parser(image_settings, setup_type)
                    used_generator = create_vtk_generator(image_settings, used_parser, detector_type, setup_type)
                    t, r, s = create_transforms(np.array(info.get("cameraTranslation")), Rotation.from_rotvec(info.get("cameraRotation"), degrees=False), transforms_type, distance)
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
    # generate_virtual_images()
    run_default_calibration("test")
