import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from python.C_imagesGeneration import ImageGenerationSettings
from python.D_tagsDetection import perform_detection
from python.E_visualization import two_parameter_relation_show, show_trajectory
from python.models.transformsParser.kalmanParser import SimpleKalmanFilterParser
from python.settings import test_camera_matrix
from python.utils import copy_camera_profile_info
from python.workflow_utils import create_image_generation_settings, create_transforms, create_aruco_detector, \
    create_parser, create_vtk_generator, run_default_calibration


def experiments_test():
    image_settings = create_image_generation_settings('aruco', '')
    profiles_to_use = ["x_y", "x_z", "x_rx", "x_ry", "x_rz", "traj_1", "traj_2", "traj_3"]
    profiles_transforms = [create_transforms(np.array([0, 0, 0]), Rotation.from_rotvec([0, 0, 0]), setup) for setup in profiles_to_use]

    used_detector = create_aruco_detector(profiles_to_use[0], image_settings, True)
    used_transform = create_parser(image_settings, 'cube')
    used_generator = create_vtk_generator(image_settings, used_transform, 'aruco', 'cube', test_camera_matrix)

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
