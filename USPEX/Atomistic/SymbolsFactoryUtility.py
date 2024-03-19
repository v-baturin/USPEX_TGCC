import logging
import numpy as np
from collections import Counter


class SymbolsFactory:
    def __init__(self, presetSymbols: list):
        self.presetSymbols = presetSymbols

    def getRandomSymbols(self, n=1):
        return np.random.choice(self.presetSymbols, size=n, replace=True)


class SymbolsFactoryUtility:
    def __init__(self, **factories):
        self.symbolsFactories = {k: SymbolsFactory(v) for k, v in factories.items()}
        self.allFactoriesTrivial = not (True in [len(f.presetSymbols) > 1 for f in self.symbolsFactories.values()])

    def getRandomSymComposition(self, factoryComposition):
        randMolSyms = [self.symbolsFactories[k].getRandomSymbols(v) for k, v in factoryComposition.items()]
        randMolSymsFlat = [mol for molGroup in randMolSyms for mol in molGroup]
        return Counter(randMolSymsFlat)
