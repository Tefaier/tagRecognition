from main import *
from python.E_visualization import _mask_by_success, _make_x_axis_info
from python.models.transformsParser.kalmanParser import SimpleKalmanFilterParser
from python.models.transformsParser.transformsParser import TransformsParser

if __name__ == "__main__":
    profiles = build_profiles_strings(True, True, True, ["traj_1", "traj_2"])
    all_info = read_info(profiles)
    info = get_info_part(all_info, build_profile(True, True, True, "traj_1"), {"aruco3": False, "parser": "simple"})
    mask = _mask_by_success(info, None)
    translation = False
    axis = 'x'
    x = _make_x_axis_info(info, mask, True, 't')
    y_real_t = [info["realT"][index][axis_to_index(axis)] for index in mask]
    y_real_r = [get_rotation_euler(info["realR"][index], axis, True) for index in mask]
    y_detected_t = [info["detectedT"][index] for index in mask]
    y_detected_r = [info["detectedR"][index] for index in mask]
    info_size = len(x)
    x = np.array(x)
    y_real_t = np.array(y_real_t).reshape((info_size, 1))
    y_real_r = np.array(y_real_r).reshape((info_size, 1))
    y_detected_t = np.array(y_detected_t).reshape((info_size, 3))
    y_detected_r = np.array(y_detected_r).reshape((info_size, 3))

    y_filtered_t = []
    y_filtered_r = []
    filter = SimpleKalmanFilterParser(
        TransformsParser([np.array([0, 0, 0])], [Rotation.from_rotvec([0, 0, 0])], [0]),
        0.5,
        False,
        False
    )

    for i in range(0, len(x)):
        t, r = filter.get_parent_transform([y_detected_t[i]], [Rotation.from_rotvec(y_detected_r[i], degrees=False)], [0], x[i])
        y_filtered_t.append(t)
        y_filtered_r.append(r)
    y_filtered_t = np.array(y_filtered_t)
    y_filtered_r = np.array(y_filtered_r)

    plt.figure(figsize=(12, 5))
    plt.plot(x, y_real_t, '.', label="Real")
    plt.scatter(x, [y_detected_t[index][axis_to_index(axis)] for index in range(0, y_detected_t.shape[0])], label="Detected no filter", alpha=0.8, s=10)
    plt.scatter(x, [y_filtered_t[index][axis_to_index(axis)] for index in range(0, y_filtered_t.shape[0])], label="Detected with filter", alpha=0.3, s=25)
    plt.legend()
    plt.show()
