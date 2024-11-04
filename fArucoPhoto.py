# детекция аруко меток на фото

import cv2
import cv2.aruco as aruco
import numpy as np
import math

# scipy нужен для преобразования матрицы вращения в углы Эйлера в градусах с фиксированными осями
# положительное вращение по часовой стрелке, если смотреть в положительном направлении оси
from scipy.spatial.transform import Rotation 

image_path = "exp2/WIN_20241104_12_49_54_Pro.jpg"

image = cv2.imread(image_path)
# image = cv2.resize(image, (640, 480))

dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
detectorParams = aruco.DetectorParameters()

detector = aruco.ArucoDetector(dictionary, detectorParams)

imageCopy = image.copy()

# markerCorners содержит координаты 4 углов метки
markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(image)


markerLength = 0.0525  # длина стороны маркера в метрах

# координаты углов маркера в его собственной системе координат
objPoints = np.array([[-markerLength / 2, markerLength / 2, 0],
                      [markerLength / 2, markerLength / 2, 0],
                      [markerLength / 2, -markerLength / 2, 0],
                      [-markerLength / 2, -markerLength / 2, 0]])

if markerIds is not None:
    # параметры камеры (моей камеры на ноутбуке)
    camMatrix = np.array([[804.7329058535828, 0.0, 549.3237487667773], [0.0, 802.189566021595, 293.62680986426403], [0.0, 0.0, 1.0]])

    # Коэффициенты дисторсии (моей камеры на ноутбуке)
    distCoeffs = np.array([-0.12367717208987415, 1.3006314330799533, -0.00045665885332229637, -0.028794247586331707, -2.264152794148503])  
             
    cnt = 0
    for i in range(len(markerCorners)):
        # if (markerIds[i] != 40):
        #     continue

        success, rvec, tvec = cv2.solvePnP(objPoints, markerCorners[i], camMatrix, distCoeffs)

        if success:
            cv2.drawFrameAxes(imageCopy, camMatrix, distCoeffs, rvec, tvec, markerLength * 1.5, 2)

            print(f'Метка номер {cnt}')
            print(f'Расстояние: {tvec.flatten()}') # расстояние от камеры до центра метки в метрах

            rotation_matrix, _ = cv2.Rodrigues(rvec)
            r = Rotation.from_matrix(rotation_matrix)

            euler_angles = r.as_euler('ZYX', degrees=True)
            z_angle_deg, y_angle_deg, x_angle_deg = euler_angles
            print(f"Углы поворота (в градусах): X={x_angle_deg}, Y={y_angle_deg}, Z={z_angle_deg}", end = '\n\n')

            cnt += 1

    # Подсвечивание обнаруженных маркеров
    aruco.drawDetectedMarkers(imageCopy, markerCorners, markerIds)

cv2.imshow("out", imageCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()
