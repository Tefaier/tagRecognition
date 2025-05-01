import numpy as np
from scipy.spatial.transform import Rotation

from python.models.imageGenerators.imageGenerator import ImageGenerator


class PseudoImageGenerator(ImageGenerator):
    def __init__(self):
        super().__init__()

    def generate_image_with_obj_at_transform(self, obj_translation: np.array, obj_rotation: Rotation, save_path: str) -> bool:
        return True

    def generate_images_with_obj_at_transform(self, obj_translation: np.array, obj_rotation: Rotation, save_paths: list[str]) -> bool:
        return True

    def check_transform_is_available(self, obj_translation: np.array, obj_rotation: Rotation) -> bool:
        return True
