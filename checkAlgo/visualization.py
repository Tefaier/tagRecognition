from cProfile import label

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from checkAlgo.constantsForCheck import resultFolder, analiseFile
from checkAlgo.utils import readStringOfList, getRotationEuler, axisToIndex

analizationResults = pd.read_csv(resultFolder + "/" + analiseFile)

realT = np.array(readStringOfList(analizationResults['realT']))
realR = np.array(readStringOfList(analizationResults['realR']))
detectedT = readStringOfList(analizationResults['detectedT'])
detectedR = readStringOfList(analizationResults['detectedR'])
errorT = readStringOfList(analizationResults['errorT'])
errorR = readStringOfList(analizationResults['errorR'])
isSuccess = np.array(analizationResults['isSuccess'])
usedMethod = np.array(analizationResults['method'])
successMask = np.where(isSuccess == True)

# arucoIndexes = np.where(usedMethod == arucoDetector.name)
# apriltagIndexes = np.where(usedMethod == apriltagDetector.name)

offset = 400
# show x rotation [0, 100)
def makeXAxisInfo(specificMask: np.array, isPosition: bool, xAxisPartToShow: str):
    if isPosition:
        return [realT[index][axisToIndex(xAxisPartToShow)] for index in specificMask]
    else:
        return [getRotationEuler(realR[index], xAxisPartToShow, True) for index in specificMask]

def makeYAxisInfo(specificMask: np.array, isPosition: bool, yAxisPartToShow: str):
    if isPosition:
        return [errorT[index][axisToIndex(yAxisPartToShow)] for index in specificMask]
    else:
        return [getRotationEuler(errorR[index], yAxisPartToShow, True) for index in specificMask]

def maskBySuccess(mask: np.array):
    return np.intersect1d(mask, successMask)

def initPlot(plotRow: int, plotColumn: int, plotNumber: int, plotTitle: str, plotXAxisTitle: str, plotYAxisTitle: str):
    plt.subplot(plotRow, plotColumn, plotNumber)
    plt.title(plotTitle)
    plt.xlabel(plotXAxisTitle)
    plt.ylabel(plotYAxisTitle)

def binifyInfo(x: list, y: list, bins: int):
    binEdges = np.histogram_bin_edges(x, bins)
    binMiddles = (binEdges[1:] + binEdges[:-1]) * 0.5
    return (binMiddles, np.histogram(x, binEdges, weights=y)[0])

def makeDislpay(
        plotLabel: str,
        generalMask: np.array,
        isPosition: bool,
        xAxisPartToShow: str,
        yAxisPartToShow: str,
        binsToMake: int
):
    mask = maskBySuccess(generalMask)
    xInfo = makeXAxisInfo(mask, isPosition, xAxisPartToShow)
    yInfo = makeYAxisInfo(mask, isPosition, yAxisPartToShow)

    if binsToMake != 0:
        x, y = binifyInfo(xInfo, yInfo, binsToMake)
    else:
        x = xInfo
        y = yInfo

    plt.plot(x, y, label=plotLabel)

def xRotation():
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
def yRotation():
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
def zDisplacement():
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
def yDisplacement():
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


# shows rotation error splitted by euler
def separateRotation():
    plt.title("Relation between errors in detected rotation and real rotation")
    plt.xlabel("Real rotation around x, degrees")
    plt.ylabel("Deviation of detected rotation compared to real, degrees")

    mask = np.where(errorR > -0.5)
    mask = mask[0][np.where(mask[0] < 100)]

    realRX = [getRotationEuler(rot, 'x', True) for rot in realR[mask]]
    binEdges = np.histogram_bin_edges(realRX, 10)
    binMiddles = (binEdges[1:] + binEdges[:-1]) * 0.5

    deviationsXAruco = [getRotationEuler(realR[index], 'x', True) - getRotationEuler(detectedR[index], 'x', True) for index in mask]
    deviationsYAruco = [getRotationEuler(realR[index], 'y', True) - getRotationEuler(detectedR[index], 'y', True) for index in mask]
    deviationsZAruco = [getRotationEuler(realR[index], 'z', True) - getRotationEuler(detectedR[index], 'z', True) for index in mask]

    plt.plot(
        binMiddles,
        np.histogram(realRX, binEdges, weights=deviationsXAruco)[0],
        label='x deviation of aruco'
    )
    plt.plot(
        binMiddles,
        np.histogram(realRX, binEdges, weights=deviationsYAruco)[0],
        label='y deviation of aruco'
    )
    plt.plot(
        binMiddles,
        np.histogram(realRX, binEdges, weights=deviationsZAruco)[0],
        label='z deviation of aruco'
    )
    mask += offset

    realRX = [getRotationEuler(rot, 'x', True) for rot in realR[mask]]
    binEdges = np.histogram_bin_edges(realRX, 10)
    binMiddles = (binEdges[1:] + binEdges[:-1]) * 0.5

    deviationsXApriltag = [getRotationEuler(realR[index], 'x', True) - getRotationEuler(detectedR[index], 'x', True) for index in mask]
    deviationsYApriltag = [getRotationEuler(realR[index], 'y', True) - getRotationEuler(detectedR[index], 'y', True) for index in mask]
    deviationsZApriltag = [getRotationEuler(realR[index], 'z', True) - getRotationEuler(detectedR[index], 'z', True) for index in mask]
    plt.plot(
        binMiddles,
        np.histogram(realRX, binEdges, weights=deviationsXApriltag)[0],
        label='x deviation of apriltag'
    )
    plt.plot(
        binMiddles,
        np.histogram(realRX, binEdges, weights=deviationsYApriltag)[0],
        label='y deviation of apriltag'
    )
    plt.plot(
        binMiddles,
        np.histogram(realRX, binEdges, weights=deviationsZApriltag)[0],
        label='z deviation of apriltag'
    )
    plt.legend()
    plt.show()


initPlot(2, 4, 1,
         "Relation between errors in detected rotation and real rotation, Aruco",
         "Real rotation around x, degrees",
         "Deviation of detected rotation compared to real, degrees")
makeDislpay("x", np.arange(0, 100), False, 'x', 'x', 10)
makeDislpay("y", np.arange(0, 100), False, 'x', 'y', 10)
makeDislpay("z", np.arange(0, 100), False, 'x', 'z', 10)

initPlot(2, 4, 2,
         "Relation between errors in detected rotation and real rotation, Apriltag",
         "Real rotation around x, degrees",
         "Deviation of detected rotation compared to real, degrees")
makeDislpay("x", np.arange(0, 100), False, 'x', 'x', 10)
makeDislpay("y", np.arange(0, 100), False, 'x', 'y', 10)
makeDislpay("z", np.arange(0, 100), False, 'x', 'z', 10)

initPlot(2, 4, 3,
         "Relation between errors in detected rotation and real rotation, Aruco",
         "Real rotation around y, degrees",
         "Deviation of detected rotation compared to real, degrees")
makeDislpay("x", np.arange(0, 100), False, 'y', 'x', 10)
makeDislpay("y", np.arange(0, 100), False, 'y', 'y', 10)
makeDislpay("z", np.arange(0, 100), False, 'y', 'z', 10)

initPlot(2, 4, 4,
         "Relation between errors in detected rotation and real rotation, Apriltag",
         "Real rotation around y, degrees",
         "Deviation of detected rotation compared to real, degrees")
makeDislpay("x", np.arange(0, 100), False, 'y', 'x', 10)
makeDislpay("y", np.arange(0, 100), False, 'y', 'y', 10)
makeDislpay("z", np.arange(0, 100), False, 'y', 'z', 10)

initPlot(2, 4, 5,
         "Relation between errors in detected position and real position, Aruco",
         "Real position z, meters",
         "Deviation of detected position compared to real, meters")
makeDislpay("x", np.arange(0, 100), True, 'z', 'x', 10)
makeDislpay("y", np.arange(0, 100), True, 'z', 'y', 10)
makeDislpay("z", np.arange(0, 100), True, 'z', 'z', 10)

initPlot(2, 4, 6,
         "Relation between errors in detected position and real position, Apriltag",
         "Real position z, meters",
         "Deviation of detected position compared to real, meters")
makeDislpay("x", np.arange(0, 100), True, 'z', 'x', 10)
makeDislpay("y", np.arange(0, 100), True, 'z', 'y', 10)
makeDislpay("z", np.arange(0, 100), True, 'z', 'z', 10)

initPlot(2, 4, 7,
         "Relation between errors in detected position and real position, Aruco",
         "Real position y, meters",
         "Deviation of detected position compared to real, meters")
makeDislpay("x", np.arange(0, 100), True, 'y', 'x', 10)
makeDislpay("y", np.arange(0, 100), True, 'y', 'y', 10)
makeDislpay("z", np.arange(0, 100), True, 'y', 'z', 10)

initPlot(2, 4, 8,
         "Relation between errors in detected position and real position, Apriltag",
         "Real position y, meters",
         "Deviation of detected position compared to real, meters")
makeDislpay("x", np.arange(0, 100), True, 'y', 'x', 10)
makeDislpay("y", np.arange(0, 100), True, 'y', 'y', 10)
makeDislpay("z", np.arange(0, 100), True, 'y', 'z', 10)
