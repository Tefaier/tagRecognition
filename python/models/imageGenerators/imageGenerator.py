import numpy as np
from scipy.spatial.transform import Rotation

# considers default camera orientation in image frame to be
# x-to right
# y-to down
# z-into image
class ImageGenerator:
    def __init__(self):
        pass

    def generate_image_with_obj_at_transform(self, obj_translation: np.array, obj_rotation: Rotation, save_path: str):
        pass
