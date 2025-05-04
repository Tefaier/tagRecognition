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
    create_manipulator_generator, create_aruco_detector, hand_to_eye_calibration_on_profiles, run_image_info_creation

def generate_virtual_images(calibration_profile: str):
    for detector_type in ["aruco", "apriltag"]:
        for setup_type in ["single", "cube"]:
            for transforms_type in ["x_y", "x_z", "x_rx", "x_rz", "traj_1", "traj_2"]:
                profile_str = f"virtual_{setup_type}_{detector_type}_{transforms_type}"
                copy_camera_profile_info(calibration_profile, profile_str)
                info = read_profile_json(profile_str)
                image_settings = create_image_generation_settings(detector_type, transforms_type)
                used_parser = create_parser(image_settings, setup_type)
                used_generator = create_vtk_generator(image_settings, used_parser, detector_type, setup_type, np.array(info.get("cameraMatrix"),))
                t, r, s = create_transforms(np.array(info.get("cameraTranslation")), Rotation.from_rotvec(info.get("cameraRotation"), degrees=False), transforms_type)
                generate_images(profile_str, used_generator, image_settings, t, r, s)
                create_detections(profile_str, image_settings, used_parser, detector_type, setup_type, transforms_type)

# experiment_type is one of [x_y, x_z, x_rx, x_rz, traj_1, traj_2]
def make_images_for_experiment(profile_source: str, profile_label: str, experiment_type: str, is_aruco: bool):
    detector_type = 'aruco' if is_aruco else 'apriltag'
    profile_str = f"{profile_label}_{detector_type}_{experiment_type}"
    copy_camera_profile_info(profile_source, profile_str)
    info = read_profile_json(profile_str)
    base2camera_translation = np.array(info.get("cameraTranslation"))
    base2camera_rotation = Rotation.from_rotvec(info.get("cameraRotation"), degrees=False)
    image_settings = create_image_generation_settings(detector_type, experiment_type)
    used_generator = create_manipulator_generator(base2camera_translation, base2camera_rotation)
    t, r, s = create_transforms(base2camera_translation, base2camera_rotation, experiment_type)
    generate_images(profile_str, used_generator, image_settings, t, r, s)

    used_generator.reset()
    used_generator.to_start_pose() # return to start position

if __name__ == "__main__":
    # run_image_info_creation("calibration_real")
    # generate_virtual_images("calibration_real")
    # save_camera_info("calibration_real", [[1393.53993076, 0.0, 986.86708455],
    #      [0.0, 1392.89920241, 558.27594943],
    #      [0.0, 0.0, 1.0]], [-0.00222352, -0.06004384,  0.00375605, -0.00080867, 0])
    # profiles = ["real_single_aruco_x_rx", "real_single_aruco_x_rz", "real_single_aruco_x_y", "real_single_aruco_x_z", "real_single_aruco_traj_1", "real_single_aruco_traj_2"]
    # for profile in profiles:
    #     transforms_type = [tr for tr in ["x_y", "x_z", "x_rx", "x_ry", "x_rz", "traj_1", "traj_2"] if tr in profile][0]
    #     image_settings = create_image_generation_settings("aruco", transforms_type)
    #     create_detections(profile, image_settings, create_parser(image_settings, "single"), "aruco", "single", transforms_type)
    hand_to_eye_calibration_on_profiles("calibration_real", ["real_single_aruco_x_rx", "real_single_aruco_x_rz", "real_single_aruco_x_y", "real_single_aruco_x_z", "real_single_aruco_traj_1", "real_single_aruco_traj_2"])
    # test_camera()
    # hand_to_eye_on_manipulator("calibration_real")

    # make_images_for_experiment("calibration_real", "real_cube", "x_y", is_aruco=False)
    # make_images_for_experiment("calibration_real", "real_cube", "x_z", is_aruco=False)
    # make_images_for_experiment("calibration_real", "real_cube", "x_rx", is_aruco=False)
    # make_images_for_experiment("calibration_real", "real_cube", "x_rz", is_aruco=False)
    # make_images_for_experiment("calibration_real", "real_cube", "traj_1", is_aruco=False)
    # make_images_for_experiment("calibration_real", "real_cube", "traj_2", is_aruco=False)


    # test_manipulator('192.168.56.101', 30003, res[0], res[1], is_real = False)


    # experiments_test()
    # physics_parser_test()
    # generate_virtual_images()
    # run_default_calibration("test")
