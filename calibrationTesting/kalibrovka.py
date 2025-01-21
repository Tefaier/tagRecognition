# калибровка камеры
# нужно сделать минимум 10 фоток шахматной доски на эту камеру под разными углами и с разного расстояния
# и положить в папку calibration_images

import numpy as np
import cv2
import glob

# критерий завершения алгоритма(завершается когда будет достигнуто n итераций или точность k)
# используется в cv2.cornerSubPix, которая уточняет углы клеток доски
n, k = 30, 0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, n, k)

square_size = 0.025  # Размер одной клетки в метрах
# Координаты 3D точек внутренних углов шахматной доски. Доска расположена параллельно камере, z = 0
objp = np.zeros((7*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2) * square_size

objpoints = []
imgpoints = []  # Найденные углы

# Чтение всех изображений с шахматной доской
images = glob.glob('calibration_images2/*.jpg')

for name in images:
    img = cv2.imread(name)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Поиск углов шахматной доски
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    if ret:
        objpoints.append(objp.copy())

        # уточнение найденных углов
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, (7, 7), corners2, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(50)

cv2.destroyAllWindows()

# Калибровка камеры
ret, camMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(camMatrix.tolist())
print(distCoeffs.tolist())

cv2.waitKey(0)
cv2.destroyAllWindows()
