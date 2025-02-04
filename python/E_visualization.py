from cProfile import label

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from python.constantsForCheck import resultFolder, analiseFile
from python.settings import generatedInfoFolder, detectionInfoFilename, plotsFolder
from python.utils import readStringOfList, getRotationEuler, axisToIndex, readProfileJSON, readStringOfDict

# [[profileStr,
#   tagSize,
#   arucoFamily,
#   apriltagFamily,
#   [    detectionSetting,
#        {   'method': [],
#            'realT': [],
#            'realR': [],
#            'errorT': [],
#            'errorR': [],
#            'isSuccess': [],
#            'successMask': []
#        }
#   ]
# ]]
def readInfo(profiles: list[str]):
    result = []
    for profile in profiles:
        analizationResults = pd.read_csv(f"{os.path.dirname(__file__)}/{generatedInfoFolder}/{profile}/{detectionInfoFilename}.csv")
        jsonInfo = readProfileJSON(profile)
        resultProfile = [profile, jsonInfo["tagLength"], jsonInfo["arucoFamily"], jsonInfo["apriltagFamily"]]

        realT = np.array(readStringOfList(analizationResults['realT']))
        realR = np.array(readStringOfList(analizationResults['realR']))
        errorT = readStringOfList(analizationResults['errorT'])
        errorR = readStringOfList(analizationResults['errorR'])
        isSuccess = np.array(analizationResults['isSuccess'])
        method = np.array(analizationResults['method'])
        detectionSettings = readStringOfDict(analizationResults['detectionSettings'])

        dictBySettingsEquals = {}
        for index, setting in enumerate(detectionSettings):
            dictBySettingsEquals.setdefault(tuple(sorted(setting.items())), []).append(index)
        for key, value in dictBySettingsEquals.items():
            resultProfile.append([
                detectionSettings[dictBySettingsEquals[key][0]], {}
            ])
            resultProfile[-1][1]['method'] = [method[index] for index in value]
            resultProfile[-1][1]['realT'] = np.array([realT[index] for index in value])
            resultProfile[-1][1]['realR'] = np.array([realR[index] for index in value])
            resultProfile[-1][1]['errorT'] = [errorT[index] for index in value]
            resultProfile[-1][1]['errorR'] = [errorR[index] for index in value]
            resultProfile[-1][1]['isSuccess'] = np.array([isSuccess[index] for index in value])
            resultProfile[-1][1]['successMask'] = np.where(resultProfile[-1][1]['isSuccess'] == True)
        result.append(resultProfile)
    return result

# show x rotation [0, 100)
def makeXAxisInfo(info: dict, specificMask: np.array, isTranslation: bool, xAxisPartToShow: str):
    if isTranslation:
        return [info["realT"][index][axisToIndex(xAxisPartToShow)] for index in specificMask]
    else:
        return [getRotationEuler(info["realR"][index], xAxisPartToShow, True) for index in specificMask]

def makeYAxisInfo(info: dict, specificMask: np.array, isTranslation: bool, yAxisPartToShow: str):
    if isTranslation:
        return [info["errorT"][index][axisToIndex(yAxisPartToShow)] for index in specificMask]
    else:
        return [getRotationEuler(info["errorR"][index], yAxisPartToShow, True) for index in specificMask]

def maskBySuccess(info: dict, mask: np.array):
    return np.intersect1d(mask, info["successMask"])

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
        infoRange: tuple[float, float],
        info: dict
):
    mask = maskBySuccess(info, generalMask)
    xInfo = makeXAxisInfo(info, mask, isTranslation, xAxisPartToShow)
    yInfo = makeYAxisInfo(info, mask, isTranslation, yAxisPartToShow)

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

def getInfoPart(info: list, profile: str, requiredTuple: tuple):
    profileIndex = -1
    for i in range(len(info)):
        if info[i][0] == profile:
            profileIndex = i
            break
    if profileIndex == -1: raise ValueError(f"Didn't find profile {profile}")

    dictIndex = -1
    for i in range(len(info[profileIndex][-1])):
        if info[profileIndex][-1][i][0] == requiredTuple:
            dictIndex = i
            break
    if dictIndex == -1: raise ValueError(f"Didn't find dictionary with {requiredTuple}")

    return info[profileIndex][-1][dictIndex][1]

def testRun():
    generalInfo = readInfo(["test"])
    arucoAll = getInfoPart(generalInfo, "test", ())
    fig = initFigure("Errors in detected rotation and real rotation")
    iFrom, iTo = 0, 2500
    initSubplot(1, 1, 1,
                "Aruco",
                "Real rotation around x, degrees",
                "Deviation, degrees")
    makeDislpay("x", np.arange(iFrom, iTo), False, 'x', 'x', 50, (-85, 85), arucoAll)
    makeDislpay("y", np.arange(iFrom, iTo), False, 'x', 'y', 50, (-85, 85), arucoAll)
    makeDislpay("z", np.arange(iFrom, iTo), False, 'x', 'z', 50, (-85, 85), arucoAll)
    savePlot(fig, f'{plotsFolder}/RotationX.png')

    fig = initFigure("Errors in detected rotation and real rotation")
    iFrom, iTo = 2500, 5000
    initSubplot(1, 1, 1,
                "Aruco",
                "Real rotation around y, degrees",
                "Deviation, degrees")
    makeDislpay("x", np.arange(iFrom, iTo), False, 'y', 'x', 50, (-85, 85), arucoAll)
    makeDislpay("y", np.arange(iFrom, iTo), False, 'y', 'y', 50, (-85, 85), arucoAll)
    makeDislpay("z", np.arange(iFrom, iTo), False, 'y', 'z', 50, (-85, 85), arucoAll)
    savePlot(fig, f'{plotsFolder}/RotationY.png')


'''
fig = initFigure("Errors in detected rotation and real rotation")
iFrom, iTo = 0, 2500
initSubplot(1, 2, 1,
         "Aruco",
         "Real rotation around x, degrees",
         "Deviation, degrees")
makeDislpay("x", np.arange(iFrom, iTo), False, 'x', 'x', 50, (-85, 85), generalInfo[0][-1][-1])
makeDislpay("y", np.arange(iFrom, iTo), False, 'x', 'y', 50, (-85, 85), generalInfo[0][-1][-1])
makeDislpay("z", np.arange(iFrom, iTo), False, 'x', 'z', 50, (-85, 85), generalInfo[0][-1][-1])

iFrom, iTo = iFrom + 5000, iTo + 5000
initSubplot(1, 2, 2,
         "Apriltag",
         "Real rotation around x, degrees",
         "Deviation, degrees")
makeDislpay("x", np.arange(iFrom, iTo), False, 'x', 'x', 50, (-85, 85), generalInfo[0][-1][-1])
makeDislpay("y", np.arange(iFrom, iTo), False, 'x', 'y', 50, (-85, 85), generalInfo[0][-1][-1])
makeDislpay("z", np.arange(iFrom, iTo), False, 'x', 'z', 50, (-85, 85), generalInfo[0][-1][-1])
savePlot(fig, resultFolder + '/RotationX.png')

fig = initFigure("Errors in detected rotation and real rotation")
iFrom, iTo = 2500, 5000
initSubplot(1, 2, 1,
         "Aruco",
         "Real rotation around y, degrees",
         "Deviation, degrees")
makeDislpay("x", np.arange(iFrom, iTo), False, 'y', 'x', 50, (-85, 85), generalInfo[0][-1][-1])
makeDislpay("y", np.arange(iFrom, iTo), False, 'y', 'y', 50, (-85, 85), generalInfo[0][-1][-1])
makeDislpay("z", np.arange(iFrom, iTo), False, 'y', 'z', 50, (-85, 85), generalInfo[0][-1][-1])

iFrom, iTo = iFrom + 5000, iTo + 5000
initSubplot(1, 2, 2,
         "Apriltag",
         "Real rotation around y, degrees",
         "Deviation, degrees")
makeDislpay("x", np.arange(iFrom, iTo), False, 'y', 'x', 50, (-85, 85), generalInfo[0][-1][-1])
makeDislpay("y", np.arange(iFrom, iTo), False, 'y', 'y', 50, (-85, 85), generalInfo[0][-1][-1])
makeDislpay("z", np.arange(iFrom, iTo), False, 'y', 'z', 50, (-85, 85), generalInfo[0][-1][-1])
savePlot(fig, resultFolder + '/RotationY.png')
'''
