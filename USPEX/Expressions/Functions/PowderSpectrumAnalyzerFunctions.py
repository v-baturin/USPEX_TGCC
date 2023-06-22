class PowderSpectrumAnalyzerFunctions:
    def __init__(self, utility) -> None:
        self.utility = utility

    def xraydistance(self, system):
        if 'powderSpectrumAnalyzer.xraydistance' not in system:
            self.utility.analyze(system)
        assert 'powderSpectrumAnalyzer.xraydistance' in system
        return system['powderSpectrumAnalyzer.xraydistance']

    def k(self, system):
        if 'powderSpectrumAnalyzer.k' not in system:
            self.utility.analyze(system)
        assert 'powderSpectrumAnalyzer.k' in system
        return system['powderSpectrumAnalyzer.k']

