from typing import List

class VariationOperators(object):
    def __init__(self, hybridizationTypes : List[type], mutationTypes : List[type], creationTypes : List[type]):
        self.hybridizationTypes = hybridizationTypes
        self.mutationTypes = mutationTypes
        self.creationTypes = creationTypes