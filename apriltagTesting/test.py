from typing import Any, List

import cv2
from dt_apriltags import Detector, Detection
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation

imagePath1 = "./testImages/img.png"
imagePath2 = "./testImages/tag41_12_00000.png"

# used
# https://pyimagesearch.com/2020/11/02/apriltag-with-python/
# https://matplotlib.org/stable/gallery/mplot3d/imshow3d.html#sphx-glr-gallery-mplot3d-imshow3d-py

def imshow3d(ax, img):
    xx, yy = np.meshgrid(np.linspace(0, img.shape[1] / img.shape[0], img.shape[1] + 1), np.linspace(0, 1, img.shape[0] + 1))
    zz     = np.ones((img.shape[0] + 1,img.shape[1] + 1))
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=img, shade=False)

def createDetector() -> Detector:
    return Detector(searchpath=['apriltags'],
                       families='tagStandard41h12', # tagStandard41h12 tag25h9 tag36h11
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

def createRotatedImage(path: str, reduce: int = 1, cameraRot: List[float] = None) -> str:
    if cameraRot is None:
        cameraRot = [0, 0, 0]
    image = Image.open(path)
    if (reduce != 1):
        image = image.resize([int(x / reduce) for x in image.size])
    image = np.array(image)
    if (image.shape[-1] == 4):
        image = image[:, :, :-1]

    plt.axis('square')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=-90 + cameraRot[0], azim=cameraRot[1], roll=cameraRot[2])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # expect to width be >= height ALWAYS
    plt.xlim(0, image.shape[1] / image.shape[0])
    plt.ylim(0, image.shape[1] / image.shape[0])
    imshow3d(ax, img=image / 255)
    newName = '.'.join(path.split('.')[:-1]) + "3d.png"
    plt.savefig(newName, dpi=200, bbox_inches='tight', pad_inches=0)
    return newName

def getGrayImage(path: str) -> np.ndarray:
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def analyseImage(results: tuple[list[Detection], Any] | list[Detection], image: np.ndarray):
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
        cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2)
    return image

def generateCameraTransform(distance: float, cameraRot: List[float]):
    rot1 = Rotation.from_rotvec([cameraRot[0], 0, 0], degrees=True)
    rot2 = Rotation.from_rotvec([0, 0, cameraRot[1]], degrees=True)
    comb1 = rot2 * rot1
    rot3 = Rotation.from_rotvec(comb1.apply(np.array([0, 0, 1])) * cameraRot[2], degrees=True)
    combFull = rot3 * comb1

    return (combFull.apply(np.array([0, 0, -1 * distance])), combFull)

def localTransformToGlobal(positionLocal: np.ndarray, rotLocal: Rotation, positionInLocal: List[float], rotInLocal: Rotation):
    inverted = rotLocal.inv()
    return (positionLocal + inverted.apply(positionInLocal), rotInLocal * rotLocal)

cameraRot = [88, 0, 0]
cameraPosition, cameraRotation = generateCameraTransform(1, cameraRot)
detector = createDetector()
rotatedVersion = imagePath1 if False else createRotatedImage(imagePath2, reduce=1, cameraRot=cameraRot)
imageGray = getGrayImage(rotatedVersion)
# camera_params is camera parameters of:
# fx - x focal length in pixels
# fy - y focal length in pixels
# cx - x of focal center in pixels
# cy - y of focal center in pixels
# usually fx=fy
# usually cx = image.width/2, cy = image.height/2
#
# tag_size is size of tag in terms of distance between parallel fully black sides (in meters!)
#
# results are as follows
# The coordinate system has the origin at the camera center.
# The z-axis points from the camera center out the camera lens.
# The x-axis is to the right in the image taken by the camera, and y is down.
# The tag's coordinate frame is centered at the center of the tag, with x-axis to the right, y-axis down, and z-axis into the tag.
results = detector.detect(
    imageGray,
    estimate_tag_pose=True,
    camera_params=[100, 100, imageGray.shape[1] / 2, imageGray.shape[0] / 2], # imageGray [739, 739] when launched with plt
    tag_size=1)
print(results)
for r in results:
    globalPos, globalRot = localTransformToGlobal(cameraPosition, cameraRotation, [p[0] for p in r.pose_t], Rotation.from_matrix(r.pose_R))
    print("Position: " + str(globalPos))
    print("Rotation: " + str(globalRot.as_euler('xyz', degrees=True)))

analysedImage = analyseImage(results, cv2.imread(rotatedVersion))
cv2.imshow("Image", analysedImage)
cv2.waitKey(0)