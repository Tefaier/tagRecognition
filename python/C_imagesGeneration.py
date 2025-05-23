import glob
import os
import time

import cv2.aruco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from python.models.imageGenerators.imageGenerator import ImageGenerator
from python.models.imageGenerators.manipulatorGenerator import ManipulatorGenerator
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
    is_trajectory: bool

    def __init__(
            self,
            clear_existing_images: bool,
            tagSize: float,
            isAruco: bool,
            arucoFamily: str,
            isApriltag: bool,
            apriltagFamily: str,
            is_trajectory: bool
    ):
        self.clear_existing_images = clear_existing_images
        self.tagSize = tagSize
        self.isAruco = isAruco
        self.arucoFamily = arucoFamily
        self.isApriltag = isApriltag
        self.apriltagFamily = apriltagFamily
        self.is_trajectory = is_trajectory

    def dict_version(self) -> dict:
        return {"tagSize": self.tagSize, "isAruco": self.isAruco, "arucoFamily": self.arucoFamily, "isApriltag": self.isApriltag, "apriltagFamily": self.apriltagFamily, "isTrajectory": self.is_trajectory}


def _make_output(image_names: list, name: str, translations: list, translation: list, rotations: list, rotation: list, timings: list, timing: float):
    image_names.append(f"{name}.png")
    translations.append([float(val) for val in translation])
    rotations.append([float(val) for val in rotation])
    if timings is not None:
        timings.append(timing)

def _save_profile_info(profile: str, settings: ImageGenerationSettings):
    write_info_to_profile_json(profile, settings.dict_version())

def _save_generated_info(path: str, imageNames: list, translations: list, rotations: list, timings: list, replace_info: bool):
    collected_info = pd.DataFrame.from_dict({
        "imageName": imageNames,
        "realT": translations,
        "realR": rotations
    })
    if timings is not None:
        collected_info["time"] = timings
    if replace_info or not os.path.exists(path):
        collected_info.to_csv(path, header=True, mode='w', index=False)
        return
    df = pd.read_csv(path)
    pd.concat([df, collected_info]).to_csv(path, header=True, mode='w', index=False)

def _prepare_folder(path: str, clear: bool) -> int:
    ensure_folder_exists(path)
    files = glob.glob(f"{path}/*.png")
    if clear:
        for f in files:
            os.remove(f)
        return 0
    files = [int(name.split('.')[0].split('\\')[-1].split('/')[-1]) for name in files]
    to_write_from = max(files, default=-1) + 1
    return to_write_from

def generate_images(
        profile: str,
        generator: ImageGenerator,
        settings: ImageGenerationSettings,
        translations: list[list],
        rotations: list[Rotation],
        timings: list[float] = None,
        samples: int = 1
):
    # TODO uncomment later
    # if (settings.is_trajectory and (timings is None or not len(timings) == len(translations) or not settings.clear_existing_images)):
    #     raise Exception("If you use trajectory you must clear images and have size of time arrays the same as that of transforms")
    profile_folder = f"{os.path.dirname(__file__)}/{generated_info_folder}/{profile}"
    to_write_from = 0 # _prepare_folder(f"{profile_folder}/{analyse_images_folder}", settings.clear_existing_images)
    _save_profile_info(profile, settings)

    imageNames = []
    translations_write = []
    rotations_write = []
    timings_write = [] if settings.is_trajectory else None

    p_bar = tqdm(range(len(translations) * samples), ncols=100)

    info = [] # информация о недоступных точках

    for iteration_index in range(len(translations)):
        translation = translations[iteration_index]
        rotation = rotations[iteration_index]
        success = generator.check_transform_is_available(translation, rotation)

        if not success:
            p_bar.update(samples)
            p_bar.refresh()
            info.append(list(map(float, translation)))
            continue
        
        success = generator.generate_images_with_obj_at_transform(
            translation,
            rotation,
            [f"{profile_folder}/{analyse_images_folder}/{to_write_from + iteration_index * samples + i}.png" for i in range(samples)]
        )
        
        if not success:
            p_bar.update(samples)
            p_bar.refresh()
            continue

        rotation = rotation.as_rotvec(degrees=False).tolist()
        for i in range(samples):
            _make_output(
                imageNames,
                str(to_write_from + iteration_index * samples + i),
                translations_write,
                translation,
                rotations_write,
                rotation,
                timings_write,
                timings[iteration_index] if settings.is_trajectory else 0.0
            )
        p_bar.update(samples)
        p_bar.refresh()
    p_bar.close()

    print(f'\nNot avaliable\n{info}\n')

    _save_generated_info(
        f"{profile_folder}/{image_info_filename}.csv",
        imageNames,
        translations_write,
        rotations_write,
        timings_write,
        settings.clear_existing_images
    )


def test_run():
    # images present in tagImages for resized from 354 to 450 with new pixels being white border
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
            0.1 * 450.0 / 354.0,
            0.1 * 450.0 / 354.0),
        ImageGenerationSettings(True, 0.1, True, str(cv2.aruco.DICT_5X5_100), False, "", False),
        translations,
        rotations
    )

def test_manipulator(robot_ip, robot_port, translations, rotations, is_real):
    profile = "manipulator test"

    camera_translation = np.array([0, 0, 0]) # пока что будем все измерять относительно стандартной ск
    camera_rotation = Rotation.from_rotvec([0, 0, 0], degrees=False) * Rotation.from_rotvec([0, 0, 0], degrees=False)
    gripper_to_object_translation = np.array([0, 0, 0])
    gripper_to_object_rotation = Rotation.from_rotvec([0, 0, 0], degrees=True)

    start_time = time.monotonic()
    generate_images(
        profile,
        ManipulatorGenerator(
            is_real, 
            robot_ip,
            robot_port,
            camera_translation,
            camera_rotation,
            gripper_to_object_translation,
            gripper_to_object_rotation,
            take_screenshot=False
        ),
        ImageGenerationSettings(True, 0.1, True, str(cv2.aruco.DICT_5X5_100), False, "", False),
        translations,
        rotations
    )
    end_time = time.monotonic()

    print(f"Total time taken (s): {round(end_time - start_time, 2)}")
    data = pd.read_csv(f"python/generatedInfo/{profile}/{image_info_filename}.csv")
    entries = data.shape[0]
    images = glob.glob(f"python/generatedInfo/{profile}/{analyse_images_folder}/*.png")
    print(f"{len(translations)} images were requested")
    print(f"{len(images)} images were saved as png")
    print(f"{entries} images were saved as csv entries")
