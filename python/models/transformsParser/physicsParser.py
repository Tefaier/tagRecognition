from typing import Tuple
from python.models.transformsParser.transformsParser import TransformsParser
from scipy.spatial.transform import Rotation
import numpy as np

from python.utils import rotation_to_vector


# consider using filterpy.kalman.UnscentedKalmanFilter
# or something from it kalman related

class SimpleAccelerationConstraintsParser(TransformsParser):
    child_parser: TransformsParser

    max_acc: Tuple[float, float]
    max_rot_acc: Tuple[float, float]
    max_transl_glitch: float
    max_rot_glitch: float

    last_detection_time: float
    last_detection_lifetime: float
    last_detected_translation: np.ndarray[float]
    last_detected_rotation: Rotation
    last_detected_translation_speed: np.ndarray[float]
    last_detected_rotation_speed: np.ndarray[float]  # rotvec in degrees

    flip: bool

    def __init__(
            self,
            child_parser: TransformsParser,
            maximum_acceleration_range: Tuple[float, float],  # from negative to positive (in m/sec)
            maximum_rotation_acceleration_range: Tuple[float, float],  # from negative to positive (in degrees/sec)
            accepted_translation_glitch: float,  # in m
            accepted_rotation_glitch: float,  # in degrees
            last_detection_lifetime: float,
            try_flipping_tags: bool
    ):
        super().__init__(child_parser.translations, child_parser.rotations, child_parser.ids)
        self.child_parser = child_parser

        self.max_acc = maximum_acceleration_range
        self.max_rot_acc = maximum_rotation_acceleration_range
        self.max_transl_glitch = accepted_translation_glitch
        self.max_rot_glitch = accepted_rotation_glitch

        self.last_detection_time = None
        self.last_detection_lifetime = last_detection_lifetime
        self.last_detected_translation = None
        self.last_detected_rotation = None
        self.last_detected_translation_speed = None
        self.last_detected_rotation_speed = None

        self.flip = try_flipping_tags

    def _get_mirror_rotation(self, t: np.ndarray[float], r: Rotation) -> Rotation:
        # TODO try to implement actual way
        # now it is just appr of rotating to look at camera and then again
        rotation_to_face_camera = rotation_to_vector(r.apply([0, 0, 1]), -t)
        return rotation_to_face_camera * rotation_to_face_camera * r

    def get_parent_transform(
            self,
            translations: list[np.array],
            rotations: list[Rotation],
            ids: list[int],
            time: float = None
    ) -> (np.array, np.array):
        if self.flip and self.last_detected_rotation is not None and time - self.last_detection_time > self.last_detection_lifetime:
            for i in range(0, len(ids)):
                if self.tags.get(ids[i], None) is None:
                    continue
                l_t = self.tags.get(ids[i])[0]
                l_r = self.tags.get(ids[i])[1]
                r_mirrored = self._get_mirror_rotation(translations[i], rotations[i])
                deviation_t_1 = np.linalg.norm(translations[i] - (rotations[i] * l_r.inv()).apply(l_t) - self.last_detected_translation)
                deviation_t_2 = np.linalg.norm(translations[i] - (r_mirrored * l_r.inv()).apply(l_t) - self.last_detected_translation)
                deviation_r_1 = (rotations[i] * l_r.inv() * self.last_detected_rotation.inv()).magnitude()
                deviation_r_2 = (r_mirrored * l_r.inv() * self.last_detected_rotation.inv()).magnitude()
                if deviation_t_1 > deviation_t_2 and deviation_r_1 > deviation_r_2:
                    rotations[i] = r_mirrored

        result = self.child_parser.get_parent_transform(translations, rotations, ids, time)
        if len(result[0]) == 0:
            return result

        if self.last_detection_time is None or time - self.last_detection_time > self.last_detection_lifetime:
            self.last_detection_time = time
            self.last_detected_translation_speed = None
            self.last_detected_rotation_speed = None
            self.last_detected_translation = result[0]
            self.last_detected_rotation = Rotation.from_rotvec(result[1], degrees=False)
            return result

        time_since_last = time - self.last_detection_time
        translation_change = result[0] - self.last_detected_translation
        translation_speed = translation_change / time_since_last
        rotation_change = Rotation.from_rotvec(result[1], degrees=False) * self.last_detected_rotation.inv()
        rotation_speed = rotation_change.as_rotvec(degrees=True) / time_since_last

        if self.last_detected_translation_speed is None:
            self.last_detection_time = time
            self.last_detected_translation_speed = translation_speed
            self.last_detected_rotation_speed = rotation_speed
            self.last_detected_translation = result[0]
            self.last_detected_rotation = Rotation.from_rotvec(result[1], degrees=False)
            return result

        t_acc = np.linalg.norm(translation_speed) - np.linalg.norm(self.last_detected_translation_speed)
        r_acc = np.linalg.norm(rotation_speed) - np.linalg.norm(self.last_detected_rotation_speed)
        t_passed = (0 > t_acc > self.max_acc[0]) or (0 <= t_acc < self.max_acc[1])
        r_passed = (0 > r_acc > self.max_rot_acc[0]) or (0 <= r_acc < self.max_rot_acc[1])

        if t_passed and r_passed:
            self.last_detection_time = time
            self.last_detected_translation_speed = translation_speed
            self.last_detected_rotation_speed = rotation_speed
            self.last_detected_translation = result[0]
            self.last_detected_rotation = Rotation.from_rotvec(result[1], degrees=False)
            return result

        if ((not t_passed and np.linalg.norm(translation_change) < self.max_transl_glitch) or t_passed) and ((not r_passed and np.linalg.norm(rotation_change.as_rotvec(degrees=True)) < self.max_rot_glitch ) or r_passed):
            return result

        return [], []

