import numpy as np
from scipy.spatial.transform import Rotation

from python.utils import from_global_in_local_to_global_of_local


class TransformsParser:
    tags: dict[int, list]

    def __init__(self, translations: list[np.array], rotations: list[Rotation], ids: list[int]):
        self.translations = translations
        self.rotations = rotations
        self.ids = ids
        self.tags = {}
        for i in range(0, len(ids)):
            self.tags[ids[i]] = [translations[i], rotations[i]]

    def get_parent_transform(self, translations: list[np.array], rotations: list[Rotation], ids: list[int], time: float = None) -> (np.array , np.array):
        parent_translations = []
        parent_rotations = []
        for i in range(0, len(ids)):
            if self.tags.get(ids[i], None) is None:
                continue
            local_transform = self.tags.get(ids[i])
            t, r = from_global_in_local_to_global_of_local(translations[i], rotations[i], local_transform[0], local_transform[1])
            parent_translations.append(t)
            parent_rotations.append(r)
        if len(parent_translations) == 0:
            return [], []
        parent_translations = np.array(parent_translations)
        parent_translation = parent_translations.mean(axis=0)
        parent_rotation = Rotation.concatenate(parent_rotations).mean()
        return parent_translation, parent_rotation.as_rotvec(degrees=False)

    def get_parser_dict(self) -> dict:
        return {"parser": "simple"}
