from ...DataModel.Flavour import Flavour


class PowderSpectrumAnalyzerFunctions:
    def __init__(self, utility) -> None:
        self.utility = utility

    def xraydistance(self, system: Flavour):
        if 'powderSpectrumAnalyzer.xraydistance' not in system:
            self.utility.analyze(system)
        assert 'powderSpectrumAnalyzer.xraydistance' in system
        return system['powderSpectrumAnalyzer.xraydistance']

    def k(self, system: Flavour):
        if 'powderSpectrumAnalyzer.k' not in system:
            self.utility.analyze(system)
        assert 'powderSpectrumAnalyzer.k' in system
        return system['powderSpectrumAnalyzer.k']

