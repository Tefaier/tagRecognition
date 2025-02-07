import glob
import os

import cv2.aruco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from python.models.imageGenerators.imageGenerator import ImageGenerator
from python.models.imageGenerators.vtkGenerator import VTKGenerator
from python.settings import generated_info_folder, analyse_images_folder, image_info_filename, \
    tag_images_folder, image_height, image_width, test_camera_matrix
from python.utils import deviate_transform, generate_normal_distribution_value, ensure_folder_exists, \
    write_info_to_profile_json


class ImageGenerationSettings:
    clear_existing_images: bool
    tagSize: float
    isAruco: bool
    arucoFamily: str
    isApriltag: bool
    apriltagFamily: str

    def __init__(
            self,
            clear_existing_images: bool,
            tagSize: float,
            isAruco: bool,
            arucoFamily: str,
            isApriltag: bool,
            apriltagFamily: str
    ):
        self.clear_existing_images = clear_existing_images
        self.tagSize = tagSize
        self.isAruco = isAruco
        self.arucoFamily = arucoFamily
        self.isApriltag = isApriltag
        self.apriltagFamily = apriltagFamily

    def dict_version(self) -> dict:
        return {"tagSize": self.tagSize, "isAruco": self.isAruco, "arucoFamily": self.arucoFamily, "isApriltag": self.isApriltag, "apriltagFamily": self.apriltagFamily}


def make_output(image_names: list, name: str, translations: list, translation: list, rotations: list, rotation: list):
    image_names.append(f"{name}.png")
    translations.append([float(val) for val in translation])
    rotations.append([float(val) for val in rotation])

def save_profile_info(profile: str, settings: ImageGenerationSettings):
    write_info_to_profile_json(profile, settings.dict_version())

def save_generated_info(path: str, imageNames: list, translations: list, rotations: list, replace_info: bool):
    collected_info = pd.DataFrame.from_dict({
        "imageName": imageNames,
        "realT": translations,
        "realR": rotations
    })
    if replace_info or not os.path.exists(path):
        collected_info.to_csv(path, header=True, mode='w', index=False)
        return
    df = pd.read_csv(path)
    pd.concat([df, collected_info]).to_csv(path, header=True, mode='w', index=False)

def prepare_folder(path: str, clear: bool) -> int:
    ensure_folder_exists(path)
    files = glob.glob(f"{path}/*.png")
    if clear:
        for f in files:
            os.remove(f)
        return 0
    files = [int(name.split('.')[0]) for name in files]
    to_write_from = max(files, default=-1) + 1
    return to_write_from

def generate_images(profile: str, generator: ImageGenerator, settings: ImageGenerationSettings, translations: list[list], rotations: list[Rotation]):
    to_write_from = prepare_folder(f"{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{analyse_images_folder}", settings.clear_existing_images)
    save_profile_info(profile, settings)

    imageNames = []
    translations_write = []
    rotations_write = []

    p_bar = tqdm(range(len(translations)), ncols=100)

    for iteration_index in range(len(translations)):
        generator.generate_image_with_obj_at_transform(
            translations[iteration_index],
            rotations[iteration_index],
            f"{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{analyse_images_folder}/{to_write_from + iteration_index}.png"
        )
        make_output(
            imageNames,
            str(to_write_from + iteration_index),
            translations_write,
            translations[iteration_index],
            rotations_write,
            rotations[iteration_index].as_rotvec(degrees=False).tolist()
        )
        p_bar.update()
        p_bar.refresh()

    save_generated_info(
        f"{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{image_info_filename}.csv",
        imageNames,
        translations_write,
        rotations_write,
        settings.clear_existing_images
    )


def test_run():
    translations = []
    rotations = []
    default_translation = [0.0, 0.0, 4.0]
    samples_to_get = 50

    start_stop, spots = (-85, 85), 50
    for x in np.linspace(start_stop[0], start_stop[1], spots):
        deviate_value = (start_stop[1] - start_stop[0]) / (spots * 2)
        raw_translation = default_translation
        raw_rotation = [x + 180, 0, 0]
        for i in range(0, samples_to_get):
            translation, rotation_euler = deviate_transform(raw_translation, raw_rotation,
                                                            rx=generate_normal_distribution_value(
                                                            max_deviation=deviate_value))
            rotation = Rotation.from_euler('xyz', rotation_euler, degrees=True)
            translations.append(translation)
            rotations.append(rotation)

    start_stop, spots = (-85, 85), 50
    for y in np.linspace(start_stop[0], start_stop[1], spots):
        deviate_value = (start_stop[1] - start_stop[0]) / (spots * 2)
        raw_translation = default_translation
        raw_rotation = [180, y, 0]
        for i in range(0, samples_to_get):
            translation, rotation_euler = deviate_transform(raw_translation, raw_rotation,
                                                           ry=generate_normal_distribution_value(
                                                              max_deviation=deviate_value))
            rotation = Rotation.from_euler('xyz', rotation_euler, degrees=True)
            translations.append(translation)
            rotations.append(rotation)

    generate_images(
        "test",
        VTKGenerator(
            image_width,
            image_height,
            [np.array([0, 0, 0])],
            [Rotation.from_rotvec([0, 0, 0])],
            [f'{os.path.dirname(__file__)}/{tag_images_folder}/aruco_1.png'],
            test_camera_matrix,
            0.1,
            0.1),
        ImageGenerationSettings(True, 0.1, True, str(cv2.aruco.DICT_5X5_100), False, ""),
        translations,
        rotations
    )
