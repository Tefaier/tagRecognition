from filterpy.common import Q_discrete_white_noise

from python.models.transformsParser.transformsParser import TransformsParser
from scipy.spatial.transform import Rotation
import numpy as np
from filterpy.kalman import KalmanFilter

from python.utils import rotation_to_vector, get_mirror_rotation


# consider using filterpy.kalman.UnscentedKalmanFilter
# or something from it kalman related

class SimpleKalmanFilterParser(TransformsParser):
    child_parser: TransformsParser

    last_detection_time: float
    last_detection_lifetime: float
    last_detected_translation: np.ndarray[float]
    last_detected_rotation: Rotation
    k_filter: KalmanFilter
    flip_series: dict[int, int]

    flip: bool
    filter: bool

    def __init__(
            self,
            child_parser: TransformsParser,
            last_detection_lifetime: float,
            try_flipping_tags: bool,
            use_only_best: bool
    ):
        super().__init__(child_parser.translations, child_parser.rotations, child_parser.ids)
        self.child_parser = child_parser

        self.last_detection_time = None
        self.last_detection_lifetime = last_detection_lifetime
        self.last_detected_translation = None
        self.last_detected_rotation = None
        self.k_filter = None
        self.flip_series = {}

        self.flip = try_flipping_tags
        self.filter = use_only_best

    def _create_filter(self, pos: np.ndarray[float], speed: np.ndarray[float]) -> KalmanFilter:
        f = KalmanFilter(dim_x=6, dim_z=3)
        f.alpha = 1.01
        f.x = np.concatenate([pos, speed], axis=0)
        transition = np.eye(6, 6)
        transition += np.eye(6, 6, 3)
        f.F = transition
        measurement = np.zeros((3, 6))
        measurement[[0, 1, 2], [0, 1, 2]] = 1
        f.H = measurement
        f.P *= 1000.0
        f.R = np.eye(3, 3) * 4
        return f

    def get_parent_transform(
            self,
            translations: list[np.array],
            rotations: list[Rotation],
            ids: list[int],
            time: float = None
    ) -> (np.array, np.array):
        if self.flip and self.last_detected_translation is not None and time - self.last_detection_time < self.last_detection_lifetime:
            for i in range(0, len(ids)):
                if self.tags.get(ids[i], None) is None:
                    continue
                l_t = self.tags.get(ids[i])[0]
                l_r = self.tags.get(ids[i])[1]
                r_mirrored = get_mirror_rotation(translations[i], rotations[i])
                deviation_t_1 = np.linalg.norm(translations[i] - (rotations[i] * l_r.inv()).apply(l_t) - self.last_detected_translation)
                deviation_t_2 = np.linalg.norm(translations[i] - (r_mirrored * l_r.inv()).apply(l_t) - self.last_detected_translation)
                deviation_r_1 = (rotations[i] * l_r.inv() * self.last_detected_rotation.inv()).magnitude()
                deviation_r_2 = (r_mirrored * l_r.inv() * self.last_detected_rotation.inv()).magnitude()
                # print(time)
                # print(self.last_detected_rotation.as_rotvec(degrees=True))
                # print((rotations[i] * l_r.inv()).as_rotvec(degrees=True))
                # print((r_mirrored * l_r.inv()).as_rotvec(degrees=True))
                current_flip_series = self.flip_series.get(ids[i], 0)
                if (deviation_t_1 > deviation_t_2 - 0.0001 and deviation_r_1 > deviation_r_2 + np.deg2rad(10 + current_flip_series * 5)) or (deviation_t_1 > deviation_t_2 + 0.02 + current_flip_series * 0.01 and deviation_r_1 > deviation_r_2 - 0.001):
                    rotations[i] = r_mirrored
                    self.flip_series[ids[i]] = current_flip_series + 1
                else:
                    self.flip_series[ids[i]] = 0

        if self.filter:
            best_index = -1
            best_angle = 0
            for i in range(0, len(ids)):
                if self.tags.get(ids[i], None) is None:
                    continue
                # print(rotations[i].as_euler('xyz', degrees=True))
                angle = rotation_to_vector(rotations[i].apply([0, 0, 1]), -translations[i]).magnitude()
                if best_index == -1 or best_angle > angle:
                    best_index = i
                    best_angle = angle
            if not best_index == -1:
                translations = [translations[best_index]]
                rotations = [rotations[best_index]]
                ids = [ids[best_index]]

        result = self.child_parser.get_parent_transform(translations, rotations, ids, time)
        if len(result[0]) == 0:
            return result

        self.last_detected_rotation = Rotation.from_rotvec(result[1], degrees=False)

        if self.last_detection_time is None or time - self.last_detection_time > self.last_detection_lifetime:
            self.k_filter = None
            self.last_detection_time = time
            self.last_detected_translation = result[0]
            return result

        time_shift = time - self.last_detection_time

        if self.k_filter is None:
            self.k_filter = self._create_filter(result[0], (result[0] - self.last_detected_translation) / time_shift)
            self.last_detection_time = time
            self.last_detected_translation = result[0]
            return result

        self.k_filter.predict(Q=Q_discrete_white_noise(dim=2, dt=time_shift, var=0.1, block_size=3, order_by_dim=False))
        self.k_filter.update(result[0])
        self.last_detection_time = time
        self.last_detected_translation = ((self.k_filter.x)[:3]).copy()
        return result




