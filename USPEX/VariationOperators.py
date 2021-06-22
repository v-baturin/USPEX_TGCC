"""
USPEX.Common.VariationOperators
===============================

Class describing a collection of types of variation operators

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

from typing import List

class VariationOperators(object):
    """
    Class describing a collection of types of variation operators.
    """
    def __init__(self, hybridizationTypes: List[type], mutationTypes: List[type],
                 creationTypes: List[type], seedsType: type = None):
        """
        Initializes the class.


        """
        self.hybridizationTypes = hybridizationTypes
        self.mutationTypes = mutationTypes
        self.creationTypes = creationTypes
        self.seedsType = seedsType
