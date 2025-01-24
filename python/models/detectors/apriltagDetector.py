import numpy as np
import cv2
from dt_apriltags import Detector, Detection
from scipy.spatial.transform import Rotation

from python.models.detectors.detector import TagDetector
from python.utils import getGrayImage

# considers tag orientation on image to be (rectify applied)
# x-to right
# y-to down
# z-from viewer
class ApriltagDetector(TagDetector):
    # fx - x focal length in pixels
    # fy - y focal length in pixels
    # cx - x of focal center in pixels
    # cy - y of focal center in pixels
    def __init__(self, cameraMatrix: np.ndarray, distortionCoefficients: np.ndarray, tagFamily: str = "tag36h11", name: str = 'apriltag'):
        super().__init__(name, cameraMatrix, distortionCoefficients)
        self.fx = cameraMatrix[0, 0]
        self.fy = cameraMatrix[1, 1]
        self.cx = cameraMatrix[0, 2]
        self.cy = cameraMatrix[1, 2]
        self.detector = Detector(
            searchpath=['apriltags'],
            families=tagFamily,  # tagStandard41h12 tag25h9 tag36h11
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0)

    def rectifyRotation(self, rotation: Rotation):
        extraRotation = Rotation.from_rotvec(rotation.apply([180, 0, 0]), degrees=True)
        return extraRotation * rotation

    def detect(self, image: np.ndarray, tagLength: float) -> (list, list, list):
        imageGray = getGrayImage(image)
        imageGray = cv2.undistort(imageGray, self.cameraMatrix, self.distortionCoefficients, None)
        results = self.detector.detect(
            imageGray,
            estimate_tag_pose=True,
            camera_params=[self.fx, self.fy, self.cx, self.cy],
            tag_size=tagLength)
        ids = []
        rotations = []
        translations = []
        for r in results:
            ids.append(r.tag_id)
            rotation = self.rectifyRotation(Rotation.from_matrix(r.pose_R))
            rotations.append(rotation.as_rotvec(degrees=False))
            translations.append([val[0] for val in r.pose_t])
        return translations, rotations, ids