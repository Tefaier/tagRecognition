import numpy as np
from scipy.spatial.transform import Rotation

from python.models.imageGenerators.imageGenerator import ImageGenerator


class PseudoImageGenerator(ImageGenerator):
    def __init__(self, filter_answer: list[str] = None):
        self.filter = filter_answer
        super().__init__()

    def generate_image_with_obj_at_transform(self, obj_translation: np.array, obj_rotation: Rotation, save_path: str) -> bool:
        if self.filter is not None:
            return save_path in self.filter
        return True

    def generate_images_with_obj_at_transform(self, obj_translation: np.array, obj_rotation: Rotation, save_paths: list[str]) -> bool:
        if self.filter is not None:
            for save_path in save_paths:
                if not save_path in self.filter:
                    return False
            return True
        return True

    def check_transform_is_available(self, obj_translation: np.array, obj_rotation: Rotation) -> bool:
        return True
