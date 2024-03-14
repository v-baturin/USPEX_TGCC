import logging
import numpy as np
from collections import Counter

class MoleculeFactory:
    def __init__(self, presetMolSymbols: list):
        self.presetMolSymbols = presetMolSymbols

    def getRandomMolSymbols(self, n=1):
        return np.random.choice(self.presetMolSymbols, size=n, replace=True)

class MoleculesFactoryUtility:
    def __init__(self, presetMoleculesForFactories: dict):
        self.moleculesFactories = {k: MoleculeFactory(v) for k, v in presetMoleculesForFactories.items()}

    def getRandomMolComposition(self, factoryComposition):
        randMolSyms = [self.moleculesFactories[k].getRandomMolSymbols(v) for k, v in factoryComposition.items()]
        randMolSymsFlat = [mol for molGroup in randMolSyms for mol in molGroup]
        return Counter(randMolSymsFlat)
