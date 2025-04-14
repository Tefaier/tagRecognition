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
    change_base2gripper_to_camera2object, write_info_to_profile_json
from python.workflow_utils import run_default_calibration, create_image_generation_settings, create_parser, \
    create_vtk_generator, create_transforms, create_detections, hand_to_eye_calibration, camera_calibration, \
    create_manipulator_generator, create_aruco_detector


def experiments_test():
    image_settings = create_image_generation_settings('aruco', '')
    profiles_to_use = ["x_y", "x_z", "x_rx", "x_ry", "x_rz", "traj_1", "traj_2", "traj_3"]
    profiles_transforms = [create_transforms(np.array([0, 0, 0]), Rotation.from_rotvec([0, 0, 0]), setup) for setup in profiles_to_use]

    used_detector = create_aruco_detector(profiles_to_use[0], image_settings, True)
    used_transform = create_parser(image_settings, 'cube')
    used_generator = create_vtk_generator(image_settings, used_transform, 'aruco', 'cube')

    run_default_calibration('calibration')
    for profile in profiles_to_use:
        copy_camera_profile_info('calibration', profile)

    for i in range(len(profiles_to_use) - 3):
        # generate_images(profiles_to_use[i], used_generator, image_settings, profiles_transforms[i][0], profiles_transforms[i][1])
        perform_detection(profiles_to_use[i], used_detector, used_transform, True)
    image_settings = create_image_generation_settings('aruco', 'traj')
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

    used_detector = create_aruco_detector(profiles_to_use[0], image_settings, True)
    used_transform = create_parser(image_settings, 'cube')

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


# cameraMatrix as list 3x3, distortionCoefficients as list 5
def save_camera_info(profile: str, cameraMatrix: list[list[float]], distortionCoefficients: list[float]):
    # TODO 1 - save camera intrinsic parameters or get them with calibration
    write_info_to_profile_json(profile, {"cameraMatrix": cameraMatrix, "distortionCoefficients": distortionCoefficients})

def camera_calibration_on_manipulator(profile: str):
    camera_calibration(profile, False, np.array([0, 0, 0]), Rotation.from_rotvec([0, 0, 0], degrees=True))

def hand_to_eye_on_manipulator(profile: str):
    # TODO 2 - manually calculate approximate base2camera transform and adjust generation strategy in function if needed
    hand_to_eye_calibration(profile, False, np.array([0, 0, 0]), Rotation.from_rotvec([0, 0, 0], degrees=True))

# experiment_type is one of [x_y, x_z, x_rx, x_ry, x_rz, traj_1, traj_2]
def make_images_for_experiment(profile_source: str, profile_label: str, experiment_type: str, is_aruco: bool):
    # TODO 3 - check all experiments and adjust their function so that all positions are successfully generated
    # TODO 4 - find wait timing fall all of experiments, either shared or individual
    detector_type = 'aruco' if is_aruco else 'apriltag'
    profile_str = f"{profile_label}_{detector_type}_{experiment_type}"
    copy_camera_profile_info(profile_source, profile_str)
    info = read_profile_json(profile_str)
    image_settings = create_image_generation_settings(detector_type, experiment_type)
    used_generator = create_manipulator_generator(np.array(info.get("cameraTranslation")), Rotation.from_rotvec(info.get("cameraRotation"), degrees=False))
    t, r, s = create_transforms(np.array(info.get("cameraTranslation")), Rotation.from_rotvec(info.get("cameraRotation"), degrees=False), experiment_type)
    generate_images(profile_str, used_generator, image_settings, t, r, s)

def test_exp_cube():
    # n = 8
    t = [
        [0.25, -0.135, 0.3],  # A
        [0.55, -0.135, 0.3],  # B
        [0.55,  0.135, 0.3],  # C
        [0.25,  0.135, 0.3],  # D
        [0.25,  0.135, 0.5],  # H
        [0.55,  0.135, 0.5],  # G
        [0.55, -0.135, 0.5],  # F
        [0.25, -0.135, 0.5],  # E
        [0.25, -0.135, 0.3]   # A
    ]
    r = [
        Rotation.from_rotvec([10, -5, 5], degrees=True) * Rotation.from_rotvec([0, 0, 0], degrees=True),
        Rotation.from_rotvec([0, 5, 0], degrees=True) * Rotation.from_rotvec([0, 0, 0], degrees=True),
        Rotation.from_rotvec([5, 10, -5], degrees=True) * Rotation.from_rotvec([0, 0, 0], degrees=True),
        Rotation.from_rotvec([5, -10, 5], degrees=True) * Rotation.from_rotvec([0, 0, 0], degrees=True),
        Rotation.from_rotvec([10, 5, 5], degrees=True) * Rotation.from_rotvec([0, 0, 0], degrees=True),
        Rotation.from_rotvec([-10, -5, -5], degrees=True) * Rotation.from_rotvec([0, 0, 0], degrees=True),
        Rotation.from_rotvec([10, -5, 10], degrees=True) * Rotation.from_rotvec([0, 0, 0], degrees=True),
        Rotation.from_rotvec([10, -10, 5], degrees=True) * Rotation.from_rotvec([0, 0, 0], degrees=True),
        Rotation.from_rotvec([5, -5, 5], degrees=True) * Rotation.from_rotvec([0, 0, 0], degrees=True)
    ]
    return [t, r]

if __name__ == "__main__":
    # calibrationTest()
    # handEyeTest()
    # imagesGenerationTest()
    # tagsDetectionTest()

    #test_aruco_cube()

    # res = x_y_experiment(0.5, 5)
    # res = test_exp_cube()    
    # res = simple_trajectory_experiment(0.8)
    # test_manipulator('192.168.56.101', 30003, res[0], res[1])
    # experiments_test()
    # physics_parser_test()
    # generate_virtual_images()
    run_default_calibration("test")
