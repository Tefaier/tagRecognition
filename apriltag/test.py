from typing import Any

import cv2
import apriltag
import numpy as np
import matplotlib as plt

imagePath = "/testImages"

# used
# https://pyimagesearch.com/2020/11/02/apriltag-with-python/

def createDetector() -> apriltag.Detector:
    options = apriltag.DetectorOptions(families="tagStandard41h12")  # recommended standard
    return apriltag.Detector(options)

def createRotatedImage(path: str) -> str:
    print(path)

def getGrayImage(path: str) -> np.ndarray:
    image = cv2.imread(path)
    cv2.rotatedRectangleIntersection()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def analyseImage(results: tuple[list[apriltag.Detection], Any] | list[apriltag.Detection], image: np.ndarray):
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


detector = createDetector()
rotatedVersion = createRotatedImage(imagePath)
image = getGrayImage(rotatedVersion)
results = detector.detect(image)
analyseImage(results, image)
cv2.imshow("Image", image)
cv2.waitKey(0)