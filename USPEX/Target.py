"""
USPEX.Common.Target
===================

Class describing target space

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
"""

import logging

from types import SimpleNamespace
from typing import List, NamedTuple
from copy import copy, deepcopy


logger = logging.getLogger(__name__)


class TargetType(NamedTuple):
    utilities : List[type]
    hybridizations : List[type]
    mutations : List[type]
    creations : List[type]
    seeds : type


class Target(object):
    """
    Target space is one of the most important concepts of global search algorithms. This space is formed by all the
    systems among which we perform our search and closely tied to our search. This class contains configuration of
    such space, parameters of what are we searching for, list of systems already studied in the search, current result
    of the search and tools to wisely create new systems for the search within this space.

    :ivar varOperators:
        list of all variation operators for this target space. Each element is an implementation of
        :class:`~USPEX.Common.VarOperator.VarOperator` interface. For default implementation this list is empty.
    :ivar hybridizations:
        list of variation operators with two parents for this target space.
        For default implementation this list is empty.
    :ivar mutations:
        list of variation operators with one parent for this target space.
        For default implementation this list is empty.
    :ivar creations:
        list of variation operators with no parents for this target space.
        For default implementation this list is empty.
    """

    knownTargetTypes = {}

    @classmethod
    def registerTarget(cls, name: str, utilities: List[type], hybridizations: List[type], mutations: List[type],
                       creations: List[type], seeds: type = None):
        """
        Register the target as known target.

        :type name: str
        :param name: target name.
        :type utilities: list
        :param utilities: list of types of utilities.
        :type hybridizations: list
        :param hybridizations: list of types of hybridization operators.
        :type mutations: list
        :param mutations: list of types of mutation operators.
        :type creations: list
        :param creations: list of types of mutation operators.
        :type seeds: type
        :param seeds: type of Seeds operator.
        """
        assert name not in cls.knownTargetTypes
        cls.knownTargetTypes[name] = TargetType(utilities=utilities, hybridizations=hybridizations,
                                                mutations=mutations, creations=creations, seeds=seeds)


    def __init__(self, type: str, **kwargs):
        """
        Initializes the class.

        :type type: str
        :param type: target type.
        :type kwargs: dict
        :param kwargs: parameters for initializing config.
        """
        self.name = type
        targetTypes = self.knownTargetTypes[type]
        utilities = {}
        for untilityType in targetTypes.utilities:
            name = untilityType.__name__[0].lower() + untilityType.__name__[1:]
            try:
                utilities[name] = untilityType(**kwargs[name]) if name in kwargs else untilityType()
            except TypeError as e:
                logger.debug("Utility 'SpectrumAnalyzer' lacks required spectrum data and wont be used.")
                logger.debug(e)
        self.utilities = SimpleNamespace(**utilities)

        self.hybridizations = []
        for hybridizationType in targetTypes.hybridizations:
            name = hybridizationType.__name__[0].lower() + hybridizationType.__name__[1:]
            params = kwargs[name] if name in kwargs else {}
            try:
                self.hybridizations.append(hybridizationType(self.utilities, **params))
            except RuntimeError as e:
                logger.info(e)
            except Exception as e:
                logger.error(e, exc_info=True)

        self.mutations = []
        for mutationType in targetTypes.mutations:
            name = mutationType.__name__[0].lower() + mutationType.__name__[1:]
            params = kwargs[name] if name in kwargs else {}
            try:
                self.mutations.append(mutationType(self.utilities, **params))
            except RuntimeError as e:
                logger.info(e)
            except Exception as e:
                logger.error(e, exc_info=True)

        self.creations = []
        for creationType in targetTypes.creations:
            name = creationType.__name__[0].lower() + creationType.__name__[1:]
            params = kwargs[name] if name in kwargs else {}
            try:
                self.creations.append(creationType(self.utilities, **params))
            except RuntimeError as e:
                logger.info(e)
            except Exception as e:
                logger.error(e, exc_info=True)

        seedsType = targetTypes.seeds
        if seedsType is not None:
            name = seedsType.__name__[0].lower() + seedsType.__name__[1:]
            params = kwargs[name] if name in kwargs else {}
            self.seeds = seedsType(self.utilities, **params)
        else:
            self.seeds = None

        self.variationOperators = self.hybridizations + self.mutations + self.creations

    def __copy__(self):
        other = Target.__new__(Target)
        other.utilities = self.utilities
        other.hybridizations = None
        other.mutations = None
        other.creations = None
        other.variationOperators = None
        return other
