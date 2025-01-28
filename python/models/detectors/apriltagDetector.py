import numpy as np
import cv2
from dt_apriltags import Detector, Detection
from scipy.spatial.transform import Rotation

from python.models.detectors.detector import TagDetector
from python.utils import getGrayImage

class ApriltagSettings:
    def __init__(self, nthreads: int = 1, quad_decimate: float = 1.0, quad_sigma: float = 0.0, refine_edges: int = 1, decode_sharpening: float = 0.25):
        self.nthreads = nthreads
        self.quad_decimate = quad_decimate
        self.quad_sigma = quad_sigma
        self.refine_edges = refine_edges
        self.decode_sharpening = decode_sharpening

    def dictVersion(self) -> dict:
        return {
            "nthreads": self.nthreads,
            "quad_decimate": self.quad_decimate,
            "quad_sigma": self.quad_sigma,
            "refine_edges": self.refine_edges,
            "refine_edges": self.refine_edges,
        }

# considers tag orientation on image to be (rectify applied)
# x-to right
# y-to down
# z-from viewer
class ApriltagDetector(TagDetector):
    # fx - x focal length in pixels
    # fy - y focal length in pixels
    # cx - x of focal center in pixels
    # cy - y of focal center in pixels
    def __init__(self, cameraMatrix: np.ndarray, distortionCoefficients: np.ndarray, tagLength: float, settings: ApriltagSettings, tagFamily: str = "tag36h11", name: str = 'apriltag'):
        super().__init__(name, cameraMatrix, distortionCoefficients)
        self.fx = cameraMatrix[0, 0]
        self.fy = cameraMatrix[1, 1]
        self.cx = cameraMatrix[0, 2]
        self.cy = cameraMatrix[1, 2]
        self.objPoints = np.array([[-tagLength / 2, tagLength / 2, 0],
                                   [tagLength / 2, tagLength / 2, 0],
                                   [tagLength / 2, -tagLength / 2, 0],
                                   [-tagLength / 2, -tagLength / 2, 0]], dtype=np.float32)
        self.tagLength = tagLength
        self.settings = settings
        self.detector = Detector(
            searchpath=['apriltags'],
            families=tagFamily,  # tagStandard41h12 tag25h9 tag36h11
            nthreads=settings.nthreads,
            quad_decimate=settings.quad_decimate,
            quad_sigma=settings.quad_sigma,
            refine_edges=settings.refine_edges,
            decode_sharpening=settings.decode_sharpening,
            debug=0)

    def rectifyRotation(self, rotation: Rotation):
        extraRotation = Rotation.from_rotvec(rotation.apply([180, 0, 0]), degrees=True)
        return rotation * extraRotation

    def detectObjectPoints(self, image: np.ndarray) -> (list, list):
        imageGray = getGrayImage(image)
        results = self.detector.detect(imageGray)
        markerCorners = [r.corners for r in results]
        return self.objPoints.copy(), markerCorners[0].reshape((4, 1, 2)) if len(markerCorners) > 0 else None

    def detect(self, image: np.ndarray) -> (list, list, list):
        imageGray = getGrayImage(image)
        imageGray = cv2.undistort(imageGray, self.cameraMatrix, self.distortionCoefficients, None)
        results = self.detector.detect(
            imageGray,
            estimate_tag_pose=True,
            camera_params=[self.fx, self.fy, self.cx, self.cy],
            tag_size=self.tagLength)
        ids = []
        rotations = []
        translations = []
        for r in results:
            ids.append(r.tag_id)
            rotation = self.rectifyRotation(Rotation.from_matrix(r.pose_R))
            rotations.append(rotation.as_rotvec(degrees=False))
            translations.append([val[0] for val in r.pose_t])
        return translations, rotations, ids

    def detectorSettings(self) -> dict:
        return self.settings.dictVersion()

