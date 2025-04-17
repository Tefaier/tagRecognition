import array

import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
# import math
import socket
# import pyautogui

from python.models.imageGenerators.imageGenerator import ImageGenerator
from python.utils import from_local_to_global, from_global_in_local_to_global_of_local


class ManipulatorGenerator(ImageGenerator):
    is_real: bool
    count_request = 0
    camera_translation: np.array
    camera_rotation: Rotation
    object_translation_local_to_gripper: Rotation
    object_rotation_local_to_gripper: Rotation

    def __init__(self, is_real, robotIP: str, REALTIME_PORT: int, camera_translation: np.array, camera_rotation: Rotation, object_translation_local_to_gripper: np.array, object_rotation_local_to_gripper: Rotation, camera_port: int = 0, take_screenshot: bool = False):
        super().__init__()

        self.is_real = is_real
        self.robot_ip = robotIP
        self.robot_port = REALTIME_PORT
        self.current_pos = None
        self.last_joints_pos = [0, -3.14/2, 0, -3.14/2, 0, 0]

        self.listener = JointStateListener()

        self.camera_translation = camera_translation
        self.camera_rotation = camera_rotation
        self.object_translation_local_to_gripper = object_translation_local_to_gripper
        self.object_rotation_local_to_gripper = object_rotation_local_to_gripper

        if take_screenshot:
            class ScreenshotCamera:
                def read(self):
                    pass
                    # return pyautogui.screenshot(), True
            self.camera = ScreenshotCamera()
        else:
            self.camera = cv2.VideoCapture(camera_port)

    def reset(self):
        self.count_request = 0

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

    def _send_cached_first_move_command(self) -> bool:
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
                if self.count_request < 2:
                    self.count_request += 1
                    time.sleep(10)
                else:
                    time.sleep(3)
                return True
            else:
                timeout = 20
                start_time = time.monotonic()
                while time.monotonic() - start_time < timeout:
                    js = self.listener.listen_once(3)
                    if js is None:
                        continue
                    if js.velocity == array.array('d', [0, 0, 0, 0, 0, 0]) and 1 <= time.monotonic() - start_time:
                        if self.last_joints_pos is None:
                            return True
                        if self.last_joints_pos != js.position:
                            self.last_joints_pos = js.position
                            return True
                        else:
                            return False
                return False

        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def _make_move_command(self, t: np.array, r: np.array):
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


class JointStateListener(Node):
    def __init__(self):
        rclpy.init(args=None)
        super().__init__('joint_state_listener')
        self.joint_state = None
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            1
        )

    def listener_callback(self, msg):
        self.joint_state = msg
        # self.get_logger().info("Получено сообщение /joint_states")

    def listen_once(self, timeout=0.5, listen_time=0.1):
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=listen_time)
            if self.joint_state is not None:
                break
        return self.joint_state

    def __del__(self):
        self.destroy_node()
        rclpy.shutdown()
