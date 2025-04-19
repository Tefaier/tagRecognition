import array

import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
# import math
import socket
import pyautogui

from python.models.imageGenerators.imageGenerator import ImageGenerator
from python.utils import from_local_to_global, from_global_in_local_to_global_of_local


class ManipulatorGenerator(ImageGenerator):
    is_real: bool
    count_request = True
    camera_translation: np.array
    camera_rotation: Rotation
    object_translation_local_to_gripper: Rotation
    object_rotation_local_to_gripper: Rotation

    def __init__(self, is_real, robot_ip: str, robot_port: int, camera_translation: np.array, camera_rotation: Rotation, object_translation_local_to_gripper: np.array, object_rotation_local_to_gripper: Rotation, camera_port: int = 0, take_screenshot: bool = False):
        super().__init__()

        self.is_real = is_real
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.current_pos = None
        self.last_joints_pos = [0, -3.14/2, 0, -3.14/2, 0, 0]

        self.camera_translation = camera_translation
        self.camera_rotation = camera_rotation
        self.object_translation_local_to_gripper = object_translation_local_to_gripper
        self.object_rotation_local_to_gripper = object_rotation_local_to_gripper

        if take_screenshot:
            class ScreenshotCamera:
                def read(self):
                    return pyautogui.screenshot(), True
            self.camera = ScreenshotCamera()
        else:
            self.camera = cv2.VideoCapture(camera_port)
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 100)  # Set brightness (default)
            self.camera.set(cv2.CAP_PROP_CONTRAST, 100)  # Set contrast (default)
            self.camera.set(cv2.CAP_PROP_SATURATION, -1)  # Set saturation (default)
            self.camera.set(cv2.CAP_PROP_SHARPNESS, 100)  # Set sharpness (default)
            self.camera.set(cv2.CAP_PROP_GAIN, -1)  # Set gain (default)
            self.camera.set(cv2.CAP_PROP_AUTO_WB, -1)  # Enable auto white balance
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1)  # Enable auto exposure
            self.camera.set(cv2.CAP_PROP_EXPOSURE, 100)  # Set exposure
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
            self.camera.set(cv2.CAP_PROP_FOCUS, -1)  # Set focus (default)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set image width to 1920
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set image height to 1080

    def reset(self):
        self.count_request = True

    def generate_image_with_obj_at_transform(self, obj_translation: np.array, obj_rotation: Rotation, save_path: str) -> bool:
        t, r = self._convert_from_camera_to_gripper(obj_translation, obj_rotation)
        r = r.as_rotvec(degrees=False)
        success = self._send_cached_move_command(t, r)
        if not success: return False
        success, image = self.camera.read()
        if not success: return False
        cv2.imwrite(save_path, image)
        return True

    def generate_images_with_obj_at_transform(self, obj_translation: np.array, obj_rotation: Rotation, save_paths: list[str]) -> bool:
        t, r = self._convert_from_camera_to_gripper(obj_translation, obj_rotation)
        r = r.as_rotvec(degrees=False)
        success = self._send_cached_move_command(t, r)
        if not success: return False
        images = []
        for _ in save_paths:
            success, image = self.camera.read()
            if not success: return False  # camera read fail - then all requested images are dropped
            images.append(image)
        for index, path in enumerate(save_paths):
            cv2.imwrite(path, images[index])
        return True

    def check_transform_is_available(self, obj_translation: np.array, obj_rotation: Rotation) -> bool:
        t, r = self._convert_from_camera_to_gripper(obj_translation, obj_rotation)
        r = r.as_rotvec(degrees=False)
        return self._send_cached_move_command(t, r)


    def _convert_from_camera_to_gripper(self, obj_translation: np.array, obj_rotation: Rotation) -> (np.array, Rotation):
        return from_global_in_local_to_global_of_local(
            *from_local_to_global(self.camera_translation, self.camera_rotation, obj_translation, obj_rotation),
            self.object_translation_local_to_gripper,
            self.object_rotation_local_to_gripper
        )

    def _send_cached_move_command(self, t: np.array, r: np.array) -> bool:
        if self.current_pos is not None and np.max(np.abs(self.current_pos - np.concatenate([t, r]))) < 1e-8: return True
        success = self._send_command_with_response(self._make_move_command(t, r))
        if success: self.current_pos = np.concatenate([t, r])
        return success

    def to_start_pose(self) -> bool:
        success = self._send_command_with_response(self._make_first_move_command())
        return success

    def _send_command_with_response(self, command: str) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.robot_ip, self.robot_port))

            command = command + "\n"

            s.sendall(command.encode('utf-8'))

            s.close()

            if self.is_real:
                if self.count_request:
                    self.count_request = False
                    time.sleep(10)
                else:
                    time.sleep(3)
                return True

        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def _make_move_command(self, t: np.array, r: np.array):
        print(t)
        print(r)
        urscript_command = f'''
def myProg():
    target_pose = p[{t[0]}, {t[1]}, {t[2]}, {r[0]}, {r[1]}, {r[2]}]
    success = is_within_safety_limits(target_pose)

    if success:
        movej(target_pose, a=1.2, v=0.5, r = 0)
        textmsg("ok")
        textmsg(target_pose)
        textmsg(get_inverse_kin(target_pose))
    else:
        textmsg("bad")
        textmsg(target_pose)
        textmsg(get_inverse_kin(target_pose))
    end
end
myProg()
'''
        return urscript_command

    def _make_first_move_command(self):
        urscript_command = f'''
def myProg():
    target_pose = [0, -3.14/2, 0, -3.14/2, 0, 0]
    success = is_within_safety_limits(target_pose)

    if success:
        movej(target_pose, a=1.2, v=0.5, r = 0)
        textmsg("ok first")
        textmsg(target_pose)
    else:
        textmsg("bad first")
        textmsg(target_pose)
    end
end
myProg()
'''
        return urscript_command
