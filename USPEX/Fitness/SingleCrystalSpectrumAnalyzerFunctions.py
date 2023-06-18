class SingleCrystalSpectrumAnalyzerFunctions:
    def __init__(self, utility) -> None:
        self.utility = utility

    def xraydistance(self, system):
        if 'singleCrystalSpectrumAnalyzer.xraydistance' not in system:
            self.utility.analyze(system)
        assert 'singleCrystalSpectrumAnalyzer.xraydistance' in system
        return system['singleCrystalSpectrumAnalyzer.xraydistance']

