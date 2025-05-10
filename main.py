from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import ast

from python.C_imagesGeneration import test_run as imagesGenerationTest, generate_images
from python.E_visualization import *
from python.E_visualization import _make_y_axis_info, _mask_by_success
from python.utils import read_profile_json, generate_random_norm_vector, copy_camera_profile_info, \
    change_base2gripper_to_camera2object, write_info_to_profile_json, find_matching_dict_index, reorder_list
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

def build_profile(is_real: bool, is_single: bool, is_aruco: bool, traj: str) -> str:
    v1 = "real" if is_real else "virtual"
    v2 = "single" if is_single else "cube"
    v3 = "aruco" if is_aruco else "apriltag"
    v4 = traj
    return f"{v1}_{v2}_{v3}_{v4}"

def build_profiles_strings(is_real: bool = None, is_single: bool = None, is_aruco: bool = None, traj: list[str] = None) -> list[str]:
    profiles = []
    iter_1 = ["real", "virtual"]
    if is_real is not None:
        iter_1 = ["real"] if is_real else ["virtual"]
    iter_2 = ["single", "cube"]
    if is_single is not None:
        iter_2 = ["single"] if is_single else ["cube"]
    iter_3 = ["aruco", "apriltag"]
    if is_aruco is not None:
        iter_3 = ["aruco"] if is_aruco else ["apriltag"]
    iter_4 = ["x_y", "x_z", "x_rx", "x_ry", "x_rz", "traj_1", "traj_2"]
    if traj is not None:
        iter_4 = traj
    for v1 in iter_1:
        for v2 in iter_2:
            for v3 in iter_3:
                for v4 in iter_4:
                    profiles.append(f"{v1}_{v2}_{v3}_{v4}")
    return profiles

def compute_info(
        info: dict,
        y_is_translation: bool = False,
        y_axis_part_to_show: str = 'all',
        merge_entries_method: Merge_methods = Merge_methods.mean,
        missing_percentage: bool = False
):
    if missing_percentage:
        return 100 * np.sum(np.where(info["isSuccess"], 0, 1)) / len(info["isSuccess"])
    mask = _mask_by_success(info, None)
    y = _make_y_axis_info(info, mask, y_is_translation, y_axis_part_to_show)
    if merge_entries_method is Merge_methods.mean:
        y = y
    elif merge_entries_method is Merge_methods.divergence_squared:
        y_mean = np.mean(y, axis=0)
        y = (y - y_mean) ** 2
    elif merge_entries_method is Merge_methods.divergence_module:
        y_mean = np.mean(y, axis=0)
        y = abs(y - y_mean)
    else:
        raise ValueError("Unsupported Merge_methods used")
    return np.mean(y, axis=0)

# returns metrics by missed, translation, rotation
def compute_metric(info: dict) -> Tuple[float, float, float]:
    return (
        compute_info(info, missing_percentage=True),
        compute_info(info, True, 'all', Merge_methods.mean),
        compute_info(info, False, 'all', Merge_methods.mean)
    )

def get_metric_indexes(profiles: list[dict], required_dict: dict) -> list[int]:
    indexes = []
    for i in range(len(profiles)):
        success = True
        for key, value in required_dict.items():
            if not profiles[i].get(key) == value:
                success = False
                break
        if success:
            indexes.append(i)
    return indexes

if __name__ == "__main__":
    pass

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
    # hand_to_eye_calibration_on_profiles("calibration_real", ["real_single_aruco_x_rx", "real_single_aruco_x_rz", "real_single_aruco_x_y", "real_single_aruco_x_z", "real_single_aruco_traj_1", "real_single_aruco_traj_2"])
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


'''
This can be used to extract some exact group of information

    profiles = build_profiles_strings(None, None, None, ["traj_1", "traj_2"])
    all_info = read_info(profiles)
    info = get_info_part(all_info, build_profile(True, True, True, "traj_1"), {"aruco3": False, "parser": "simple"})
'''

'''
This can be used to make different graphs

    profiles = build_profiles_strings(None, None, None, ["traj_1", "traj_2"])
    all_info = read_info(profiles)
    info1 = get_info_part(all_info, build_profile(True, True, True, "traj_1"), {"aruco3": False, "parser": "simple"})
    info2 = get_info_part(all_info, build_profile(True, False, True, "traj_1"), {"aruco3": False, "parser": "simple"})
    info3 = get_info_part(all_info, build_profile(True, True, True, "traj_1"), {"aruco3": True, "parser": "simple"})
    info4 = get_info_part(all_info, build_profile(True, False, True, "traj_1"), {"aruco3": True, "parser": "simple"})
    info5 = get_info_part(all_info, build_profile(True, True, False, "traj_1"), {"parser": "simple"})
    info6 = get_info_part(all_info, build_profile(True, False, False, "traj_1"), {"parser": "simple"})
    info7 = get_info_part(all_info, build_profile(False, True, True, "traj_1"), {"aruco3": False, "parser": "simple"})
    info8 = get_info_part(all_info, build_profile(False, False, True, "traj_1"), {"aruco3": False, "parser": "simple"})
    info9 = get_info_part(all_info, build_profile(False, True, True, "traj_1"), {"aruco3": True, "parser": "simple"})
    info10 = get_info_part(all_info, build_profile(False, False, True, "traj_1"), {"aruco3": True, "parser": "simple"})
    info11 = get_info_part(all_info, build_profile(False, True, False, "traj_1"), {"parser": "simple"})
    info12 = get_info_part(all_info, build_profile(False, False, False, "traj_1"), {"parser": "simple"})
    fig = init_figure(f"Plot of traj_1", size=(16, 11))
    init_subplot(2,1,1,'Relation by mean error - real images',f'Time, s',f'Rotation error, degrees')
    make_display_by_threshold("total - aruco - single", None, True, "t", False, "all", 0.2, info1, Merge_methods.mean, 0)
    make_display_by_threshold("total - aruco - cube", None, True, "t", False, "all", 0.2, info2, Merge_methods.mean, 0)
    make_display_by_threshold("total - aruco3 - single", None, True, "t", False, "all", 0.2, info3, Merge_methods.mean, 0)
    make_display_by_threshold("total - aruco3 - cube", None, True, "t", False, "all", 0.2, info4, Merge_methods.mean, 0)
    make_display_by_threshold("total - apriltag - single", None, True, "t", False, "all", 0.2, info5, Merge_methods.mean, 0)
    make_display_by_threshold("total - apriltag - cube", None, True, "t", False, "all", 0.2, info6, Merge_methods.mean, 0)
    init_subplot(2, 1, 2, 'Relation by mean error - virtual images', f'Time, s', f'Rotation error, degrees')
    make_display_by_threshold("total - aruco - single", None, True, "t", False, "all", 0.2, info7, Merge_methods.mean,0)
    make_display_by_threshold("total - aruco - cube", None, True, "t", False, "all", 0.2, info8, Merge_methods.mean, 0)
    make_display_by_threshold("total - aruco3 - single", None, True, "t", False, "all", 0.2, info9, Merge_methods.mean,0)
    make_display_by_threshold("total - aruco3 - cube", None, True, "t", False, "all", 0.2, info10, Merge_methods.mean, 0)
    make_display_by_threshold("total - apriltag - single", None, True, "t", False, "all", 0.2, info11,Merge_methods.mean, 0)
    make_display_by_threshold("total - apriltag - cube", None, True, "t", False, "all", 0.2, info12, Merge_methods.mean,0)

    save_plot(fig,f'{plots_folder}/error in traj_1 by rotation.png')
'''

'''
    traj = "traj_2"
    is_translation = True

    info1 = get_info_part(all_info, build_profile(True, True, True, traj), {"aruco3": False, "parser": "simple"})
    info2 = get_info_part(all_info, build_profile(True, False, True, traj), {"aruco3": False, "parser": "simple"})
    info5 = get_info_part(all_info, build_profile(True, True, False, traj), {"parser": "simple"})
    info6 = get_info_part(all_info, build_profile(True, False, False, traj), {"parser": "simple"})
    info7 = get_info_part(all_info, build_profile(False, True, True, traj), {"aruco3": False, "parser": "simple"})
    info8 = get_info_part(all_info, build_profile(False, False, True, traj), {"aruco3": False, "parser": "simple"})
    info11 = get_info_part(all_info, build_profile(False, True, False, traj), {"parser": "simple"})
    info12 = get_info_part(all_info, build_profile(False, False, False, traj), {"parser": "simple"})
    fig = init_figure(f"Plot of {traj}", size=(16, 11))
    init_subplot(2, 1, 1, 'Relation by mean module of deviation of error - real images', f'Time, s', f'Module of deviation, {"m" if is_translation else "degrees"}')
    make_display_by_threshold("total - aruco - single", None, True, "t", is_translation, "all", 0.2, info1, Merge_methods.divergence_module,0)
    make_display_by_threshold("total - aruco - cube", None, True, "t", is_translation, "all", 0.2, info2, Merge_methods.divergence_module, 0)
    make_display_by_threshold("total - apriltag - single", None, True, "t", is_translation, "all", 0.2, info5,Merge_methods.divergence_module, 0)
    make_display_by_threshold("total - apriltag - cube", None, True, "t", is_translation, "all", 0.2, info6, Merge_methods.divergence_module,0)
    init_subplot(2, 1, 2, 'Relation by mean module of deviation of error - virtual images', f'Time, s', f'Rotation error, degrees')
    make_display_by_threshold("total - aruco - single", None, True, "t", is_translation, "all", 0.2, info7, Merge_methods.divergence_module,0)
    make_display_by_threshold("total - aruco - cube", None, True, "t", is_translation, "all", 0.2, info8, Merge_methods.divergence_module, 0)
    make_display_by_threshold("total - apriltag - single", None, True, "t", is_translation, "all", 0.2, info11,Merge_methods.divergence_module, 0)
    make_display_by_threshold("total - apriltag - cube", None, True, "t", is_translation, "all", 0.2, info12, Merge_methods.divergence_module,0)

    save_plot(fig, f'{plots_folder}/error in {traj} by {"translation" if is_translation else "rotation"}.png')
'''

'''
This can be used to get total value some kind of

    result = compute_info(info, True, "x", Merge_methods.divergence_module)
    result = compute_info(info, True, "y", Merge_methods.divergence_module)
    result = compute_info(info, True, "z", Merge_methods.divergence_module)
    result = compute_info(info, True, "all", Merge_methods.divergence_module)
    
    print(compute_info(get_info_part(all_info, build_profile(True, True, True, "traj_1"), {"aruco3": False, "parser": "simple"}), missing_percentage=True))
    print(compute_info(get_info_part(all_info, build_profile(True, True, True, "traj_1"), {"aruco3": True, "parser": "simple"}), missing_percentage=True))
    print(compute_info(get_info_part(all_info, build_profile(True, True, False, "traj_1"), {"parser": "simple"}), missing_percentage=True))
'''


'''
This can be used to create metrics info and save it to file

    profiles = build_profiles_strings(None, None, None, ["traj_1", "traj_2"])
    all_info = read_info(profiles)

    dicts = []
    results = []
    dict = {}
    show_dict = {}
    for v1 in ["real", "virtual"]:
        show_dict["environment"] = v1
        for v2 in ["single", "cube"]:
            show_dict["composition"] = v2
            for v3 in ["aruco", "apriltag"]:
                show_dict["method"] = v3
                for v4 in ["traj_1", "traj_2"]:
                    show_dict["experiment"] = v4
                    for v5 in [False, True] if v3 == "aruco" else [None]:
                        if v5 is None:
                            if dict.__contains__("aruco3"):
                                dict.pop("aruco3")
                        else:
                            dict["aruco3"] = v5
                        for v6 in ["simple", "kalman"]:
                            dict["parser"] = v6
                            if v6 == "kalman":
                                for v7 in [False, True]:
                                    dict["flip"] = v7
                                    for v8 in [False] if v2 == "single" else [False, True]:
                                        dict["filter"] = v8
                                        info = get_info_part(all_info, build_profile(v1 == "real", v2 == "single", v3 == "aruco", v4), dict)
                                        dicts.append(dict | show_dict)
                                        results.append(compute_metric(info))
                            else:
                                if dict.__contains__("flip"):
                                    dict.pop("flip")
                                if dict.__contains__("filter"):
                                    dict.pop("filter")
                                info = get_info_part(all_info, build_profile(v1 == "real", v2 == "single", v3 == "aruco", v4), dict)
                                dicts.append(dict | show_dict)
                                results.append(compute_metric(info))
    print(dicts)
    df = pd.DataFrame()
    df["info"] = dicts
    df["missed"] = [metrics[0] for metrics in results]
    df["translation"] = [metrics[1] for metrics in results]
    df["rotation"] = [metrics[2] for metrics in results]
    df.to_csv("metrics.csv")
'''

'''
This can be used to read metrics info

    df = pd.read_csv("metrics.csv")
    profiles = [ast.literal_eval(d) for d in df["info"]]
    missed = df["missed"].values
    translations = df["translation"].values
    rotations = df["rotation"].values
'''

'''
This can be used to analyze metrics by max min

    df = pd.read_csv("metrics.csv")
    profiles = [ast.literal_eval(d) for d in df["info"]]
    missed = df["missed"].values
    translations = df["translation"].values
    rotations = df["rotation"].values
    indexes = get_metric_indexes(profiles, {"experiment": "traj_2", "environment": "virtual"})
    p_f = [profiles[i] for i in indexes]
    m_f = np.array(missed[indexes])
    t_f = np.array(translations[indexes])
    r_f = np.array(rotations[indexes])
    min_i_by_m = np.argmin(m_f)
    min_i_by_t = np.argmin(t_f)
    min_i_by_r = np.argmin(r_f)
    max_i_by_m = np.argmax(m_f)
    max_i_by_t = np.argmax(t_f)
    max_i_by_r = np.argmax(r_f)
    print(f"Min by missed with value {m_f[min_i_by_m]} at profile: {p_f[min_i_by_m]}")
    print(f"Max by missed with value {m_f[max_i_by_m]} at profile: {p_f[max_i_by_m]}")
    print(f"Min by transl with value {t_f[min_i_by_t]}/{r_f[min_i_by_t]} at profile: {p_f[min_i_by_t]}")
    print(f"Max by transl with value {t_f[max_i_by_t]}/{r_f[max_i_by_t]} at profile: {p_f[max_i_by_t]}")
    print(f"Min by rotati with value {t_f[min_i_by_r]}/{r_f[min_i_by_r]} at profile: {p_f[min_i_by_r]}")
    print(f"Max by rotati with value {t_f[max_i_by_r]}/{r_f[max_i_by_r]} at profile: {p_f[max_i_by_r]}")
'''

'''
This can be used to make metrics comparison

    df = pd.read_csv("metrics.csv")
    profiles = [ast.literal_eval(d) for d in df["info"]]
    missed = df["missed"].values
    translations = df["translation"].values
    rotations = df["rotation"].values
    indexes = get_metric_indexes(profiles, {"experiment": "traj_2", "environment": "real", "parser": "kalman"})
    p_f_all = [profiles[i] for i in indexes]
    m_f_all = np.array(missed[indexes])
    t_f_all = np.array(translations[indexes])
    r_f_all = np.array(rotations[indexes])
    indexes_from = get_metric_indexes(p_f_all, {"flip": False, "filter": False})
    p_f_from = [p_f_all[i] for i in indexes_from]
    m_f_from = np.array(m_f_all[indexes_from])
    t_f_from = np.array(t_f_all[indexes_from])
    r_f_from = np.array(r_f_all[indexes_from])
    indexes_to = get_metric_indexes(p_f_all, {"flip": False, "filter": True})
    p_f_to = [p_f_all[i] for i in indexes_to]
    m_f_to = np.array(m_f_all[indexes_to])
    t_f_to = np.array(t_f_all[indexes_to])
    r_f_to = np.array(r_f_all[indexes_to])
    shuffle_list = [find_matching_dict_index(p_f_from, p, ["flip", "filter"]) for p in p_f_to]
    p_f_to = reorder_list(p_f_to, shuffle_list)
    m_f_to = np.array(reorder_list(m_f_to, shuffle_list))
    t_f_to = np.array(reorder_list(t_f_to, shuffle_list))
    r_f_to = np.array(reorder_list(r_f_to, shuffle_list))

    m_change = np.mean(m_f_to - m_f_from)
    t_change = np.mean(t_f_to - t_f_from)
    r_change = np.mean(r_f_to - r_f_from)
    print(f"Mean change by missed {m_change}")
    print(f"Mean change by transl {t_change}")
    print(f"Mean change by rotati {r_change}")
'''
