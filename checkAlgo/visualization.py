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
def makeXAxisInfo(specificMask: np.array, isTranslation: bool, xAxisPartToShow: str):
    if isTranslation:
        return [realT[index][axisToIndex(xAxisPartToShow)] for index in specificMask]
    else:
        return [getRotationEuler(realR[index], xAxisPartToShow, True) for index in specificMask]

def makeYAxisInfo(specificMask: np.array, isTranslation: bool, yAxisPartToShow: str):
    if isTranslation:
        return [errorT[index][axisToIndex(yAxisPartToShow)] for index in specificMask]
    else:
        return [getRotationEuler(errorR[index], yAxisPartToShow, True) for index in specificMask]

def maskBySuccess(mask: np.array):
    return np.intersect1d(mask, successMask)

def initSubplot(plotRow: int, plotColumn: int, plotNumber: int, plotTitle: str, plotXAxisTitle: str, plotYAxisTitle: str):
    plt.subplot(plotRow, plotColumn, plotNumber)
    plt.title(plotTitle)
    plt.xlabel(plotXAxisTitle)
    plt.ylabel(plotYAxisTitle)

def binifyInfo(x: list, y: list, bins: int, infoRange: tuple[float, float]):
    binEdges = None
    if (infoRange is None):
        binEdges = np.histogram_bin_edges(x, bins)
    else:
        infoPoints = np.linspace(infoRange[0], infoRange[1], bins)
        offset = infoPoints[1] - infoPoints[0]
        binEdges = np.concat([[min(min(x), infoPoints[0] - 0.5 * offset)], (infoPoints[1:] + infoPoints[:-1]) * 0.5, [max(max(x), infoPoints[-1] + 0.5 * offset)]], axis=0)

    binMiddles = (binEdges[1:] + binEdges[:-1]) * 0.5
    binCounters = np.histogram(x, binEdges)[0]
    return (binMiddles, np.divide(np.histogram(x, binEdges, weights=y)[0], binCounters))

def makeDislpay(
        plotLabel: str,
        generalMask: np.array,
        isTranslation: bool,
        xAxisPartToShow: str,
        yAxisPartToShow: str,
        binsToMake: int,
        infoRange: tuple[float, float]
):
    mask = maskBySuccess(generalMask)
    xInfo = makeXAxisInfo(mask, isTranslation, xAxisPartToShow)
    yInfo = makeYAxisInfo(mask, isTranslation, yAxisPartToShow)

    if binsToMake != 0:
        x, y = binifyInfo(xInfo, yInfo, binsToMake, infoRange)
    else:
        x = xInfo
        y = yInfo

    plt.plot(x, y, label=plotLabel)
    plt.legend(loc=1)

def initFigure(
        title: str,
        size: tuple = None
):
    if size is None: size = (14, 4)
    fig = plt.figure(figsize=size)
    plt.suptitle(title)
    return fig


def savePlot(
        figure: plt.figure,
        path: str,
):
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(path, dpi='figure', bbox_inches='tight', pad_inches=0.2, edgecolor='blue')
    plt.close(figure)


fig = initFigure("Errors in detected rotation and real rotation")
iFrom, iTo = 0, 3000
initSubplot(1, 2, 1,
         "Aruco",
         "Real rotation around x, degrees",
         "Deviation, degrees")
makeDislpay("x", np.arange(iFrom, iTo), False, 'x', 'x', 60, (-85, 85))
makeDislpay("y", np.arange(iFrom, iTo), False, 'x', 'y', 60, (-85, 85))
makeDislpay("z", np.arange(iFrom, iTo), False, 'x', 'z', 60, (-85, 85))

iFrom, iTo = iFrom + 11000, iTo + 11000
initSubplot(1, 2, 2,
         "Apriltag",
         "Real rotation around x, degrees",
         "Deviation, degrees")
makeDislpay("x", np.arange(iFrom, iTo), False, 'x', 'x', 60, (-85, 85))
makeDislpay("y", np.arange(iFrom, iTo), False, 'x', 'y', 60, (-85, 85))
makeDislpay("z", np.arange(iFrom, iTo), False, 'x', 'z', 60, (-85, 85))
savePlot(fig, resultFolder + '/RotationX.png')

fig = initFigure("Errors in detected rotation and real rotation")
iFrom, iTo = 3000, 6000
initSubplot(1, 2, 1,
         "Aruco",
         "Real rotation around y, degrees",
         "Deviation, degrees")
makeDislpay("x", np.arange(iFrom, iTo), False, 'y', 'x', 60, (-85, 85))
makeDislpay("y", np.arange(iFrom, iTo), False, 'y', 'y', 60, (-85, 85))
makeDislpay("z", np.arange(iFrom, iTo), False, 'y', 'z', 60, (-85, 85))

iFrom, iTo = iFrom + 11000, iTo + 11000
initSubplot(1, 2, 2,
         "Apriltag",
         "Real rotation around y, degrees",
         "Deviation, degrees")
makeDislpay("x", np.arange(iFrom, iTo), False, 'y', 'x', 60, (-85, 85))
makeDislpay("y", np.arange(iFrom, iTo), False, 'y', 'y', 60, (-85, 85))
makeDislpay("z", np.arange(iFrom, iTo), False, 'y', 'z', 60, (-85, 85))
savePlot(fig, resultFolder + '/RotationY.png')

fig = initFigure("Errors in detected translation and real translation")
iFrom, iTo = 6000, 8500
initSubplot(1, 2, 1,
         "Aruco",
         "Real translation z, meters",
         "Deviation, meters")
makeDislpay("x", np.arange(iFrom, iTo), True, 'z', 'x', 50, (0.1, 4.5))
makeDislpay("y", np.arange(iFrom, iTo), True, 'z', 'y', 50, (0.1, 4.5))
makeDislpay("z", np.arange(iFrom, iTo), True, 'z', 'z', 50, (0.1, 4.5))

iFrom, iTo = iFrom + 11000, iTo + 11000
initSubplot(1, 2, 2,
         "Apriltag",
         "Real translation z, meters",
         "Deviation, meters")
makeDislpay("x", np.arange(iFrom, iTo), True, 'z', 'x', 50, (0.1, 4.5))
makeDislpay("y", np.arange(iFrom, iTo), True, 'z', 'y', 50, (0.1, 4.5))
makeDislpay("z", np.arange(iFrom, iTo), True, 'z', 'z', 50, (0.1, 4.5))
savePlot(fig, resultFolder + '/TranslateX.png')

fig = initFigure("Errors in detected translation and real translation")
iFrom, iTo = 8500, 11000
initSubplot(1, 2, 1,
         "Aruco",
         "Real translation y, meters",
         "Deviation, meters")
makeDislpay("x", np.arange(iFrom, iTo), True, 'y', 'x', 50, (-0.4, 0.4))
makeDislpay("y", np.arange(iFrom, iTo), True, 'y', 'y', 50, (-0.4, 0.4))
makeDislpay("z", np.arange(iFrom, iTo), True, 'y', 'z', 50, (-0.4, 0.4))

iFrom, iTo = iFrom + 11000, iTo + 11000
initSubplot(1, 2, 2,
         "Apriltag",
         "Real translation y, meters",
         "Deviation, meters")
makeDislpay("x", np.arange(iFrom, iTo), True, 'y', 'x', 50, (-0.4, 0.4))
makeDislpay("y", np.arange(iFrom, iTo), True, 'y', 'y', 50, (-0.4, 0.4))
makeDislpay("z", np.arange(iFrom, iTo), True, 'y', 'z', 50, (-0.4, 0.4))
savePlot(fig, resultFolder + '/TranslateY.png')

