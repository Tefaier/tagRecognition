from scipy.spatial.transform import Rotation
import numpy as np

# WARNING
# experiments here create transforms relative to some kind of world axis
# use change_base2gripper_to_camera2object to add to it transform on top and before

def x_y_experiment(entries_per_x: int, entries_per_y: int) -> (list[list[float]], list[Rotation]):
    translations = []
    rotations = []

    for x in np.linspace(-0.25, 0.25, entries_per_x):
        for y in np.linspace(-0.27, -0.22, entries_per_y):
            translations.append([x, y, 0.55])
            rotations.append(Rotation.from_rotvec([90, 0, 0], degrees=True))
    return translations, rotations, None

def x_z_experiment(entries_per_x: int, entries_per_z: int) -> (list[list[float]], list[Rotation]):
    translations = []
    rotations = []

    for x in np.linspace(-0.25, 0.25, entries_per_x):
        for z in np.linspace(0.4, 0.55, entries_per_z):
            translations.append([x, -0.25, z])
            rotations.append(Rotation.from_rotvec([90, 0, 0], degrees=True))
    return translations, rotations, None

def x_rx_experiment(entries_per_translation: int, entries_per_rotation: int) -> (list[list[float]], list[Rotation]):
    translations = []
    rotations = []

    for x in np.linspace(-0.25, 0.25, entries_per_translation):
        for rx in np.linspace(30, 150, entries_per_rotation):
            translations.append([x, -0.25, 0.5])
            rotations.append(Rotation.from_rotvec([rx, 0, 0], degrees=True))
    return translations, rotations, None

# to delete
def x_ry_experiment(entries_per_translation: int, entries_per_rotation: int) -> (list[list[float]], list[Rotation]):
    translations = []
    rotations = []

    for x in np.linspace(-0.2, 0.2, entries_per_translation):
        for ry in np.linspace(-70, 70, entries_per_rotation):
            translations.append([x, -0.3, 0.5])
            rotations.append(Rotation.from_rotvec([90, 0, 0], degrees=True) * Rotation.from_rotvec([0, ry, 0], degrees=True))
    return translations, rotations, None

def x_rz_experiment(entries_per_translation: int, entries_per_rotation: int) -> (list[list[float]], list[Rotation]):
    translations = []
    rotations = []

    for x in np.linspace(-0.25, 0.25, entries_per_translation):
        for rz in np.linspace(-180, 180, entries_per_rotation):
            translations.append([x, -0.25, 0.5])
            rotations.append(Rotation.from_rotvec([90, 0, 0], degrees=True) * Rotation.from_rotvec([0, 0, rz], degrees=True))
    return translations, rotations, None

def simple_trajectory_experiment(frame_rate: float) -> (list[list[float]], list[Rotation], list[float]):
    translations = []
    rotations = []

    pos = -0.2
    speed = 0
    acceleration = 0.05
    frame_time = 1/frame_rate
    for t in range(0, int(3.9/frame_time)):
        speed += acceleration * frame_time * 0.5
        translations.append([pos + speed * frame_time, -0.25, 0.5])
        speed += acceleration * frame_time * 0.5
        pos = translations[-1][0]
        rotations.append(Rotation.from_rotvec([90, 0, 0], degrees=True))
    acceleration = -1
    for t in range(0, int(0.2/frame_time)):
        speed += acceleration * frame_time * 0.5
        translations.append([pos + speed * frame_time, -0.25, 0.5])
        speed += acceleration * frame_time * 0.5
        pos = translations[-1][0]
        rotations.append(Rotation.from_rotvec([90, 0, 0], degrees=True))
    acceleration = -0.05
    for t in range(0, int(3.8 / frame_time) + 1):
        speed += acceleration * frame_time * 0.5
        translations.append([pos + speed * frame_time, -0.25, 0.5])
        speed += acceleration * frame_time * 0.5
        pos = translations[-1][0]
        rotations.append(Rotation.from_rotvec([90, 0, 0], degrees=True))
    acceleration = 1
    for t in range(0, int(0.2 / frame_time)):
        speed += acceleration * frame_time * 0.5
        translations.append([pos + speed * frame_time, -0.25, 0.5])
        speed += acceleration * frame_time * 0.5
        pos = translations[-1][0]
        rotations.append(Rotation.from_rotvec([90, 0, 0], degrees=True))
    return translations, rotations, (np.arange(0, len(translations)) * frame_time).tolist()

def simple_trajectory_rotation_experiment(frame_rate: float) -> (list[list[float]], list[Rotation], list[float]):
    translations = []
    rotations = []

    pos = -0.2
    speed = 0
    acceleration = 0.05
    frame_time = 1/frame_rate
    total_time = 0
    for t in range(0, int(3/frame_time)):
        speed += acceleration * frame_time * 0.5
        translations.append([pos + speed * frame_time, -0.27 + np.sin(total_time * 0.9) * 0.05, 0.5])
        speed += acceleration * frame_time * 0.5
        pos = translations[-1][0]
        rotations.append(Rotation.from_rotvec([np.sin(total_time * 2) * 70, np.sin(total_time * 2.6 - 0.5) * 70, 0], degrees=True) * Rotation.from_rotvec([90, 0, 0], degrees=True))
        total_time += frame_time
    acceleration = -0.05
    for t in range(0, int(3.9/frame_time)):
        speed += acceleration * frame_time * 0.5
        translations.append([pos + speed * frame_time, -0.27 + np.sin(total_time * 0.9) * 0.05, 0.5])
        speed += acceleration * frame_time * 0.5
        pos = translations[-1][0]
        rotations.append(Rotation.from_rotvec([np.sin(total_time * 2) * 70, np.sin(total_time * 2.6 - 0.5) * 70, 0], degrees=True) * Rotation.from_rotvec([90, 0, 0], degrees=True))
        total_time += frame_time
    return translations, rotations, (np.arange(0, len(translations)) * frame_time).tolist()

def simple_trajectory_only_rotate_experiment() -> (list[list[float]], list[Rotation], list[float]):
    translations = []
    rotations = []

    frame_time = 1/20
    total_time = 0
    for t in range(0, int(3/frame_time)):
        translations.append([0, -0.25, 0.1])
        rotations.append(Rotation.from_rotvec([((-1.5 + total_time) / 1.5) * 90, 0, 0], degrees=True) * Rotation.from_rotvec([90, 0, 0], degrees=True))
        total_time += frame_time
    return translations, rotations, (np.arange(0, len(translations)) * frame_time).tolist()
