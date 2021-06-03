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

        :type hybridizationTypes: list
        :param hybridizationTypes: list of types of hybridization operators.
        :type mutationTypes: list
        :param mutationTypes: list of types of mutation operators.
        :type creationTypes: list
        :param creationTypes: list of types of mutation operators.
        :type seedsType: type
        :param seedsType: type of Seeds operator.

        """
        self.hybridizationTypes = hybridizationTypes
        self.mutationTypes = mutationTypes
        self.creationTypes = creationTypes
        self.seedsType = seedsType
