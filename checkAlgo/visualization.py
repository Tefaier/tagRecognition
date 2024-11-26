import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from checkAlgo.algoContainers import arucoDetector, apriltagDetector
from checkAlgo.constantsForCheck import resultFolder, analiseFile
from checkAlgo.utils import readStringOfList

analizationResults = pd.read_csv(resultFolder + "/" + analiseFile)

realT = np.array(readStringOfList(analizationResults['realT']))
realR = np.array(readStringOfList(analizationResults['realR']))
errorT = np.array(readStringOfList(analizationResults['errorT']))
errorR = np.array(readStringOfList(analizationResults['errorR']))
usedMethod = np.array(analizationResults['method'])

# arucoIndexes = np.where(usedMethod == arucoDetector.name)
# apriltagIndexes = np.where(usedMethod == apriltagDetector.name)

offset = 400
# show x rotation [0, 100)
mask = np.arange(0, 100)
plt.plot(
    [Rotation.from_rotvec(rot).as_euler('xyz', degrees=True)[0] for rot in realR[mask]],
    errorR[mask]
)
mask += offset
plt.plot(
    [Rotation.from_rotvec(rot).as_euler('xyz', degrees=True)[0] for rot in realR[mask]],
    errorR[mask]
)
plt.show()


# show y rotation [100, 200)
mask = np.arange(100, 200)
plt.plot(
    [Rotation.from_rotvec(rot).as_euler('xyz', degrees=True)[1] for rot in realR[mask]],
    errorR[mask]
)
mask += offset
plt.plot(
    [Rotation.from_rotvec(rot).as_euler('xyz', degrees=True)[1] for rot in realR[mask]],
    errorR[mask]
)
plt.show()


# show z displacement [200, 300)
mask = np.arange(200, 300)
plt.plot(
    [vec[2] for vec in realT[mask]],
    errorT[mask]
)
mask += offset
plt.plot(
    [vec[2] for vec in realT[mask]],
    errorT[mask]
)
plt.show()


# show y displacement [300, 400)
mask = np.arange(300, 400)
plt.plot(
    [vec[1] for vec in realT[mask]],
    errorT[mask]
)
mask += offset
plt.plot(
    [vec[1] for vec in realT[mask]],
    errorT[mask]
)
plt.show()

