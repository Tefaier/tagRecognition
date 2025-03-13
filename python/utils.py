import ast
import json
import math
import os
import time
from pathlib import Path
import cv2

import numpy as np
import numpy.random
from pandas import Series
from scipy.spatial.transform import Rotation

from python.settings import generated_info_folder, general_info_filename

random_generator = numpy.random.Generator(np.random.default_rng(int(time.time())).bit_generator)


def axis_to_index(axis: str):
    return 0 if axis == 'x' else (1 if axis == 'y' else 2)


def parse_rotation(rotation: list) -> Rotation:
    if len(rotation) == 0: return None
    rotation = np.array(rotation)
    if rotation.size == 9:
        return Rotation.from_matrix(rotation)
    else:
        return Rotation.from_rotvec(rotation, degrees=False)


def get_rotation_euler(rotation: list, part: str, degrees: bool = False) -> float:
    rotation = parse_rotation(rotation)
    parts = rotation.as_euler('xyz', degrees=degrees)
    return float(parts[axis_to_index(part)])


def read_string_of_list(list_of_str: list[str]) -> list[list]:
    return [ast.literal_eval(lis.replace("np.float64(", '').replace(")", '')) for lis in list_of_str]


def read_string_of_dict(list_of_str: list[str]) -> list[dict]:
    return [ast.literal_eval(lis.replace("np.float64(", '').replace(")", '')) for lis in list_of_str]


def generate_normal_distribution_value(center: float = 0, max_deviation: float = 3) -> float:
    return min(
        max(
            -max_deviation,
            random_generator.normal(loc=center, scale=max_deviation / 3, size=None)),
        max_deviation
    )


def deviate_transform(translation: list, rotation: list, px: float = 0, py: float = 0, pz: float = 0, rx: float = 0,
                      ry: float = 0, rz: float = 0) -> list[list[float]]:
    answer = [[], []]
    if px == 0 and py == 0 and pz == 0:
        answer[0] = translation
    else:
        answer[0] = [translation[0] + px, translation[1] + py, translation[2] + pz]
    if rx == 0 and ry == 0 and rz == 0:
        answer[1] = rotation
    else:
        answer[1] = [rotation[0] + rx, rotation[1] + ry, rotation[2] + rz]
    return answer


def ensure_folder_exists(relative_path: str):
    Path(relative_path).mkdir(parents=True, exist_ok=True)


def get_gray_image(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def generate_random_norm_vector() -> np.array:
    random_vector = (np.random.rand(3) * 2) - 1
    length = random_vector.dot(random_vector)
    if abs(length) < 1e-2:
        return generate_random_norm_vector()
    return random_vector / math.sqrt(length)


def _update_json(new_info: dict, path: str):
    if os.path.exists(path):
        with open(path, 'r') as f:
            dic = json.load(f)
    else:
        dic = {}
    dic.update(new_info)
    with open(path, 'w') as f:
        json.dump(dic, f)


def write_info_to_profile_json(profile: str, info: dict):
    path = f'{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{general_info_filename}.json'
    ensure_folder_exists(f'{os.path.dirname(__file__)}/{generated_info_folder}/{profile}')
    _update_json(info, path)


def read_profile_json(profile: str) -> dict:
    path = f'{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{general_info_filename}.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            dic = json.load(f)
    else:
        dic = {}
    return dic


def copy_camera_profile_info(from_profile: str, to_profile: str):
    info = read_profile_json(from_profile)
    to_leave_list = ["cameraMatrix", "distortionCoefficients", "cameraTranslation", "cameraRotation"]
    to_leave_dict = {}
    for key in to_leave_list:
        to_leave_dict[key] = info[key]
    write_info_to_profile_json(to_profile, to_leave_dict)


def from_local_to_global(parent_translation: np.array, parent_rotation: Rotation, local_translation: np.array, local_rotation: Rotation) -> (np.array, Rotation):
    global_rotation = parent_rotation * local_rotation
    global_translation = parent_translation + parent_rotation.apply(local_translation)
    return global_translation, global_rotation

def from_global_in_local_to_global_of_local(global_translation: np.array, global_rotation: Rotation, local_translation: np.array, local_rotation: Rotation) -> (np.array, Rotation):
    global_of_local_rotation = local_rotation.inv() * global_rotation
    global_of_local_translation = global_translation - global_of_local_rotation.apply(local_translation)
    return global_of_local_translation, global_of_local_rotation
