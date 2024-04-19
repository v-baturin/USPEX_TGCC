from ...DataModel.Flavour import Flavour


class SingleCrystalSpectrumAnalyzerFunctions:
    def __init__(self, utility) -> None:
        self.utility = utility

    def xraydistance(self, system: Flavour):
        if 'singleCrystalSpectrumAnalyzer.xraydistance' not in system:
            self.utility.analyze(system)
        assert 'singleCrystalSpectrumAnalyzer.xraydistance' in system
        return system['singleCrystalSpectrumAnalyzer.xraydistance']

