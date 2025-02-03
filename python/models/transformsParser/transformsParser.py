from math import degrees

import numpy as np
from scipy.spatial.transform import Rotation

class TransformsParser:
    def __init__(self, translations: list[np.array], rotations: list[Rotation], ids: list[int]):
        self.translations = translations
        self.rotations = rotations
        self.ids = ids
        self.tags = {}
        for i in range(0, len(ids)):
            self.tags[ids[i]] = [translations[i], rotations[i]]

    def getParentTransform(self, translations: list[np.array], rotations: list[Rotation], ids: list[int]) -> (list ,list):
        parentTranslations = []
        parentRotations = []
        for i in range(0, len(ids)):
            if self.tags.get(ids[i], None) is None:
                continue
            idTransform = self.tags.get(ids[i])
            parentRotations.append(rotations[i] * idTransform[1].inv())
            parentTranslations.append(translations[i] - (parentRotations[-1].apply(idTransform[0])))
        if len(parentTranslations) == 0:
            return [], []
        parentTranslations = np.array(parentTranslations)
        parentTranslation = parentTranslations.mean(axis=0)
        parentRotation = Rotation.concatenate(parentRotations).mean()
        return parentTranslation.tolist(), parentRotation.as_rotvec(degrees=False)