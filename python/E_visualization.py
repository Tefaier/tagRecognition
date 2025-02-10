import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from python.settings import generated_info_folder, detection_info_filename, plots_folder
from python.utils import read_string_of_list, get_rotation_euler, axis_to_index, read_profile_json, read_string_of_dict, \
    ensure_folder_exists


# [[profileStr,
#   tagSize,
#   arucoFamily,
#   apriltagFamily,
#   [[    detectionSetting,
#        {   'method': [],
#            'realT': [],
#            'realR': [],
#            'errorT': [],
#            'errorR': [],
#            'isSuccess': [],
#            'successMask': []
#        }
#   ],...]
# ]]
def read_info(profiles: list[str]) -> list[list]:
    result = []
    for profile in profiles:
        analyze_results = pd.read_csv(f"{os.path.dirname(__file__)}/{generated_info_folder}/{profile}/{detection_info_filename}.csv")
        json_info = read_profile_json(profile)
        result_profile = [profile, json_info["tagSize"], json_info["arucoFamily"], json_info["apriltagFamily"], []]

        realT = np.array(read_string_of_list(analyze_results['realT']))
        realR = np.array(read_string_of_list(analyze_results['realR']))
        errorT = read_string_of_list(analyze_results['errorT'])
        errorR = read_string_of_list(analyze_results['errorR'])
        isSuccess = np.array(analyze_results['isSuccess'])
        method = np.array(analyze_results['method'])
        detectionSettings = read_string_of_dict(analyze_results['detectionSettings'])

        dict_of_settings_classes = {}
        for index, setting in enumerate(detectionSettings):
            dict_of_settings_classes.setdefault(tuple(sorted(setting.items())), []).append(index)
        for key, value in dict_of_settings_classes.items():
            result_profile[-1].append([
                detectionSettings[dict_of_settings_classes[key][0]], {}
            ])
            result_profile[-1][-1][1]['method'] = [method[index] for index in value]
            result_profile[-1][-1][1]['realT'] = np.array([realT[index] for index in value])
            result_profile[-1][-1][1]['realR'] = np.array([realR[index] for index in value])
            result_profile[-1][-1][1]['errorT'] = [errorT[index] for index in value]
            result_profile[-1][-1][1]['errorR'] = [errorR[index] for index in value]
            result_profile[-1][-1][1]['isSuccess'] = np.array([isSuccess[index] for index in value])
            result_profile[-1][-1][1]['successMask'] = np.where(result_profile[-1][-1][1]['isSuccess'] == True)
        result.append(result_profile)
    return result

def make_x_axis_info(info: dict, specific_mask: np.array, is_translation: bool, x_axis_part_to_show: str):
    if is_translation:
        return [info["realT"][index][axis_to_index(x_axis_part_to_show)] for index in specific_mask]
    else:
        return [get_rotation_euler(info["realR"][index], x_axis_part_to_show, True) for index in specific_mask]

def make_y_axis_info(info: dict, specific_mask: np.array, is_translation: bool, y_axis_part_to_show: str):
    if is_translation:
        return [info["errorT"][index][axis_to_index(y_axis_part_to_show)] for index in specific_mask]
    else:
        return [get_rotation_euler(info["errorR"][index], y_axis_part_to_show, True) for index in specific_mask]

def mask_by_success(info: dict, mask: np.array):
    return np.intersect1d(mask, info["successMask"])

def init_subplot(plot_row: int, plot_column: int, plot_number: int, plot_title: str, plot_x_axis_title: str, plot_y_axis_title: str):
    plt.subplot(plot_row, plot_column, plot_number)
    plt.title(plot_title)
    plt.xlabel(plot_x_axis_title)
    plt.ylabel(plot_y_axis_title)

# TODO fix it or rather make it merge entries by similar x value
def binify_info(x: list, y: list, bins: int, info_range: tuple[float, float]):
    bin_edges = None
    if info_range is None:
        bin_edges = np.histogram_bin_edges(x, bins)
    else:
        infoPoints = np.linspace(info_range[0], info_range[1], bins)
        offset = float(infoPoints[1] - infoPoints[0])
        bin_edges = np.concat(
            (
                [min(min(x), float(infoPoints[0]) - 0.5 * offset)],
                (infoPoints[1:] + infoPoints[:-1]) * 0.5,
                [max(max(x), float(infoPoints[-1]) + 0.5 * offset)]
            ),
            axis=0
        )

    bin_middles = (bin_edges[1:] + bin_edges[:-1]) * 0.5
    bin_counters = np.histogram(x, bin_edges)[0]
    return bin_middles, np.divide(np.histogram(x, bin_edges, weights=y)[0], bin_counters)

# precision 0.1 then multiply 10 round 0 decimal
# precision 0.01 then multiply 100 round 0 decimal
# precision 0.05 then multiply 20 round 0 decimal
def make_display_by_threshold(
        plot_label: str,
        general_mask: np.array,
        is_translation: bool,
        x_axis_part_to_show: str,
        y_axis_part_to_show: str,
        merge_threshold: float,
        info: dict
):
    mask = mask_by_success(info, general_mask)
    x_info = make_x_axis_info(info, mask, is_translation, x_axis_part_to_show)
    y_info = make_y_axis_info(info, mask, is_translation, y_axis_part_to_show)
    info_size = len(x_info)
    x_info = np.array(x_info).reshape((info_size, 1))
    y_info = np.array(y_info).reshape((info_size, 1))
    for_sorting = np.concatenate([x_info, y_info], axis=1)
    for_sorting = for_sorting[for_sorting[:, 0].argsort()]
    x_info = for_sorting[:, 0]
    y_info = for_sorting[:, 1]
    x, merge_ranges = np.unique((x_info / merge_threshold).round(decimals=0) * merge_threshold, return_index = True)
    y = [np.mean(part, axis=0) for part in np.split(y_info, merge_ranges[1:])]

    plt.plot(x, y, label=plot_label)
    plt.legend(loc=1)

def make_display_by_bins(
        plot_label: str,
        general_mask: np.array,
        is_translation: bool,
        x_axis_part_to_show: str,
        y_axis_part_to_show: str,
        bins_to_make: int,
        info_range: tuple[float, float],
        info: dict
):
    mask = mask_by_success(info, general_mask)
    x_info = make_x_axis_info(info, mask, is_translation, x_axis_part_to_show)
    y_info = make_y_axis_info(info, mask, is_translation, y_axis_part_to_show)

    if bins_to_make != 0 and len(x_info) > bins_to_make * 5:
        x, y = binify_info(x_info, y_info, bins_to_make, info_range)
    else:
        sorted_by_x = [[x_info[i], y_info[i]] for i in range(len(x_info))]
        sorted_by_x.sort(key=lambda val: val[0])
        x = [val[0] for val in sorted_by_x]
        y = [val[1] for val in sorted_by_x]

    plt.plot(x, y, label=plot_label)
    plt.legend(loc=1)

def init_figure(
        title: str,
        size: tuple = None
):
    if size is None: size = (14, 4)
    fig = plt.figure(figsize=size)
    plt.suptitle(title)
    return fig


def save_plot(
        figure: plt.figure,
        path: str,
):
    ensure_folder_exists('/'.join(path.split('/')[:-1]))
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(path, dpi='figure', bbox_inches='tight', pad_inches=0.2, edgecolor='blue')
    plt.close(figure)

def get_info_part(info: list, profile: str, required_tuple: tuple):
    profile_index = -1
    for i in range(len(info)):
        if info[i][0] == profile:
            profile_index = i
            break
    if profile_index == -1: raise ValueError(f"Didn't find profile {profile}")

    dict_index = -1
    for i in range(len(info[profile_index][-1])):
        if info[profile_index][-1][i][0] == required_tuple:
            dict_index = i
            break
    if dict_index == -1: raise ValueError(f"Didn't find dictionary with {required_tuple}")

    return info[profile_index][-1][dict_index][1]

def simple_show(profiles: list[str]):
    general_info = read_info(profiles)
    for profile in general_info:
        for setting in profile[-1]:
            info = setting[1]
            mask = np.arange(0, len(info["method"]))
            fig = init_figure(f"Plot of {profile[0]} with {setting[0]}")
            init_subplot(1, 2, 1, 'Rotation', 'Real rotation x, degrees', 'Deviation, degrees')
            make_display_by_threshold("x", mask, False, 'x', 'x', 0.01, info)
            make_display_by_threshold("y", mask, False, 'x', 'y', 0.01, info)
            make_display_by_threshold("z", mask, False, 'x', 'z', 0.01, info)

            init_subplot(1, 2, 2, 'Translation', 'Real translation x, m', 'Deviation, m')
            make_display_by_threshold("x", mask, True, 'x', 'x', 0.01, info)
            make_display_by_threshold("y", mask, True, 'x', 'y', 0.01, info)
            make_display_by_threshold("z", mask, True, 'x', 'z', 0.01, info)

            save_plot(fig, f'{plots_folder}/DeviationsX.png')

def test_run():
    general_info = read_info(["test"])
    aruco_all = get_info_part(general_info, "test", ())
    fig = init_figure("Errors in detected rotation and real rotation")
    iFrom, iTo = 0, 2500
    init_subplot(1, 1, 1,
                "Aruco",
                "Real rotation around x, degrees",
                "Deviation, degrees")
    make_display_by_bins("x", np.arange(iFrom, iTo), False, 'x', 'x', 50, (-85, 85), aruco_all)
    make_display_by_bins("y", np.arange(iFrom, iTo), False, 'x', 'y', 50, (-85, 85), aruco_all)
    make_display_by_bins("z", np.arange(iFrom, iTo), False, 'x', 'z', 50, (-85, 85), aruco_all)
    save_plot(fig, f'{plots_folder}/RotationX.png')

    fig = init_figure("Errors in detected rotation and real rotation")
    iFrom, iTo = 2500, 5000
    init_subplot(1, 1, 1,
                "Aruco",
                "Real rotation around y, degrees",
                "Deviation, degrees")
    make_display_by_bins("x", np.arange(iFrom, iTo), False, 'y', 'x', 50, (-85, 85), aruco_all)
    make_display_by_bins("y", np.arange(iFrom, iTo), False, 'y', 'y', 50, (-85, 85), aruco_all)
    make_display_by_bins("z", np.arange(iFrom, iTo), False, 'y', 'z', 50, (-85, 85), aruco_all)
    save_plot(fig, f'{plots_folder}/RotationY.png')


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
