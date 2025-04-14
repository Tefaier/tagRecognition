import numpy as np
import cv2
from dt_apriltags import Detector, Detection
from scipy.spatial.transform import Rotation

from python.models.detectors.detector import TagDetector
from python.utils import get_gray_image

class ApriltagSettings:
    def __init__(
            self,
            nthreads: int = 1,
            quad_decimate: float = 1.0,
            quad_sigma: float = 0.0,
            refine_edges: int = 1,
            decode_sharpening: float = 0.25
    ):
        self.nthreads = nthreads
        self.quad_decimate = quad_decimate
        self.quad_sigma = quad_sigma
        self.refine_edges = refine_edges
        self.decode_sharpening = decode_sharpening

    def dict_version(self) -> dict:
        return {
            "nthreads": self.nthreads,
            "quad_decimate": self.quad_decimate,
            "quad_sigma": self.quad_sigma,
            "refine_edges": self.refine_edges,
            "refine_edges": self.refine_edges,
        }

# considers tag orientation on image to be (rectify applied)
# it's unknown actually, apriltag does something strange with its rotation
# x-to right
# y-to down
# z-from viewer
class ApriltagDetector(TagDetector):
    # fx - x focal length in pixels
    # fy - y focal length in pixels
    # cx - x of focal center in pixels
    # cy - y of focal center in pixels
    def __init__(
            self,
            cameraMatrix: np.ndarray,
            distortionCoefficients: np.ndarray,
            tagSize: float,
            settings: ApriltagSettings,
            tag_family: str = "tag36h11",
            name: str = 'apriltag'
    ):
        super().__init__(name, cameraMatrix, distortionCoefficients)
        self.fx = cameraMatrix[0, 0]
        self.fy = cameraMatrix[1, 1]
        self.cx = cameraMatrix[0, 2]
        self.cy = cameraMatrix[1, 2]
        self.objPoints = np.array([[-tagSize / 2, -tagSize / 2, 0],
                                   [tagSize / 2, -tagSize / 2, 0],
                                   [tagSize / 2, tagSize / 2, 0],
                                   [-tagSize / 2, tagSize / 2, 0]], dtype=np.float32)
        self.tagSize = tagSize
        self.settings = settings
        self.detector = Detector(
            searchpath=['apriltags'],
            families=tag_family,  # tagStandard41h12 tag25h9 tag36h11
            nthreads=settings.nthreads,
            quad_decimate=settings.quad_decimate,
            quad_sigma=settings.quad_sigma,
            refine_edges=settings.refine_edges,
            decode_sharpening=settings.decode_sharpening,
            debug=0)

    def rectify_rotation(self, rotation: Rotation):
        extra_rotation = Rotation.from_rotvec(rotation.apply([180, 0, 0]), degrees=True)
        # don't change multiply order!
        # for whatever reason it wants it to be so though there is no maths background
        # TODO find why
        return extra_rotation * rotation

    def detect_object_points(self, image: np.ndarray) -> (list, list):
        image_gray = get_gray_image(image)
        results = self.detector.detect(image_gray)
        marker_corners = [r.corners for r in results]
        return self.objPoints.copy(), marker_corners[0].reshape((4, 1, 2)) if len(marker_corners) > 0 else None

    def detect(self, image: np.ndarray) -> (list, list, list):
        image_gray = get_gray_image(image)
        image_gray = cv2.undistort(image_gray, self.cameraMatrix, self.distortionCoefficients, None)
        results = self.detector.detect(
            image_gray,
            estimate_tag_pose=True,
            camera_params=[self.fx, self.fy, self.cx, self.cy],
            tag_size=self.tagSize)
        ids = []
        rotations = []
        translations = []
        for r in results:
            ids.append(r.tag_id)
            rotation = self.rectify_rotation(Rotation.from_matrix(r.pose_R))
            rotations.append(rotation.as_rotvec(degrees=False))
            translations.append(np.array([val[0] for val in r.pose_t]))
        return translations, rotations, ids

    def detector_settings(self) -> dict:
        return self.settings.dict_version()
