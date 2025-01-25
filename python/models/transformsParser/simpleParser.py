from python.models.transformsParser.transformsParser import TransformsParser


class SimpleParser(TransformsParser):
    def __init__(self, translations: list, rotations: list, ids: list):
        super().__init__(translations, rotations, ids)

    def getParentTransform(self, translations: list, rotations: list, ids: list) -> (list, list):
        pass
