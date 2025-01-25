class TransformsParser:
    def __init__(self, translations: list, rotations: list, ids: list):
        self.translations = translations
        self.rotations = rotations
        self.ids = ids

    def getParentTransform(self, translations: list, rotations: list, ids: list) -> (list, list):
        pass