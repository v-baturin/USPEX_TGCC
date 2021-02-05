class Conditions:
    def __init__(self, externalPressure, volumeType):
        self.externalPressure = externalPressure
        self.volumeType = volumeType

    def calcAtomVolume(self, symbol):
        pass

    def calcCompositionVolume(self, composition):
        pass