import os

import cv2
import numpy.linalg
import vtk
import numpy as np
from scipy.spatial.transform import Rotation
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkPlaneSource
from vtkmodules.vtkIOImage import vtkImageReader2Factory, vtkPNGWriter
from vtkmodules.vtkRenderingCore import vtkTexture, vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow, \
    vtkWindowToImageFilter

from python.models.imageGenerators.imageGenerator import ImageGenerator
from python.utils import from_local_to_global, from_global_in_local_to_global_of_local


class ManipulatorGenerator(ImageGenerator):
    camera_translation: np.array
    camera_rotation: Rotation
    object_translation_local_to_gripper: Rotation
    object_rotation_local_to_gripper: Rotation

    def __init__(self, camera_translation: np.array, camera_rotation: Rotation, object_translation_local_to_gripper: np.array, object_rotation_local_to_gripper: Rotation, camera_port: int = 0):
        super().__init__()

        self.camera_translation = camera_translation
        self.camera_rotation = camera_rotation
        self.object_translation_local_to_gripper = object_translation_local_to_gripper
        self.object_rotation_local_to_gripper = object_rotation_local_to_gripper

        self.camera = cv2.VideoCapture(camera_port)

    def generate_image_with_obj_at_transform(self, obj_translation: np.array, obj_rotation: Rotation, save_path: str) -> bool:
        t, r = self._convert_from_camera_to_gripper(obj_translation, obj_rotation)
        r = r.as_rotvec(degrees=False)
        response = os.system(
f'''ros2 topic pub --once /urscript_interface/script_command std_msgs/msg/String '{{data: \"def my_prog(): 
    movej(p[{t[0]}, {t[1]}, {t[2]}, {r[0]}, {r[1]}, {r[2]}], a=1.2, v=0.25, r=0) 
end\"}}\''''
        )
        success, image = self.camera.read()
        if not success: return success
        cv2.imwrite(save_path, image)
        return True

    def generate_images_with_obj_at_transform(self, obj_translation: np.array, obj_rotation: Rotation, save_paths: list[str]) -> bool:
        t, r = self._convert_from_camera_to_gripper(obj_translation, obj_rotation)
        r = r.as_rotvec(degrees=False)
        response = os.system(
f'''ros2 topic pub --once /urscript_interface/script_command std_msgs/msg/String '{{data: \"def my_prog(): 
    movej(p[{t[0]}, {t[1]}, {t[2]}, {r[0]}, {r[1]}, {r[2]}], a=1.2, v=0.25, r=0) 
end\"}}\''''
        )
        images = []
        for path in save_paths:
            success, image = self.camera.read()
            if not success: return success
            images.append(image)
        for index, path in enumerate(save_paths):
            cv2.imwrite(path, images[index])
        return True

    def check_transform_is_available(self, obj_translation: np.array, obj_rotation: Rotation) -> bool:
        t, r = self._convert_from_camera_to_gripper(obj_translation, obj_rotation)
        r = r.as_rotvec(degrees=False)
        response = os.system(
f'''ros2 topic pub --once /urscript_interface/script_command std_msgs/msg/String '{{data: \"def my_prog():
    target_pose = p[{t[0]}, {t[1]}, {t[2]}, {r[0]}, {r[1]}, {r[2]}]
    success = is_within_safety_limits(target_pose)
    
    if success:
        movel(target_pose, a=1.2, v=0.25)
    else:
        textmsg("Нельзя переместиться в это положение") "
end\"}}\''''
        )
        return response


    def _convert_from_camera_to_gripper(self, obj_translation: np.array, obj_rotation: Rotation) -> (np.array, Rotation):
        return from_global_in_local_to_global_of_local(
            *from_local_to_global(self.camera_translation, self.camera_rotation, obj_translation, obj_rotation),
            self.object_translation_local_to_gripper,
            self.object_rotation_local_to_gripper
        )
