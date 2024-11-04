# детекция аруко меток на видео

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import math
from scipy.spatial.transform import Rotation

# если ничего не написать, будет использовать камера
video = ""

inputVideo = cv2.VideoCapture(video if video else 0)
waitTime = 10

dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
detectorParams = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, detectorParams)

markerLength = 0.0525
objPoints = np.array([[-markerLength / 2, markerLength / 2, 0],
                      [markerLength / 2, markerLength / 2, 0],
                      [markerLength / 2, -markerLength / 2, 0],
                      [-markerLength / 2, -markerLength / 2, 0]])

while True:
    ret, image = inputVideo.read()
    if not ret:
        inputVideo.release()
        inputVideo = cv2.VideoCapture(video)

        continue
    
    imageCopy = image.copy()
    
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(image)
    
    if markerIds is not None:
        camMatrix = np.array([[804.7329058535828, 0.0, 549.3237487667773], [0.0, 802.189566021595, 293.62680986426403], [0.0, 0.0, 1.0]])

        distCoeffs = np.array([-0.12367717208987415, 1.3006314330799533, -0.00045665885332229637, -0.028794247586331707, -2.264152794148503])    

        cnt = 0
        for i in range(len(markerCorners)):
            # if (markerIds[i] != 40):
            #     continue
            
            success, rvec, tvec = cv2.solvePnP(objPoints, markerCorners[i], camMatrix, distCoeffs)

            if success:
                cv2.drawFrameAxes(imageCopy, camMatrix, distCoeffs, rvec, tvec, markerLength * 1.5, 2)

                print(f'Метка номер {cnt}')
                print(f'Расстояние: {tvec.flatten()}')

                rotation_matrix, _ = cv2.Rodrigues(rvec)
                r = Rotation.from_matrix(rotation_matrix)

                euler_angles = r.as_euler('ZYX', degrees=True)
                z_angle_deg, y_angle_deg, x_angle_deg = euler_angles
                print(f"Углы поворота (в градусах): X={x_angle_deg}, Y={y_angle_deg}, Z={z_angle_deg}", end = '\n\n')

                cnt += 1
    
    cv2.imshow("out", imageCopy)
    key = cv2.waitKey(1)
    if key == 27:
        break

inputVideo.release()
cv2.destroyAllWindows()
