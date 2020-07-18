"""
USPEX.Common.Target
===================

Class describing target space

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
"""

import logging

from types import SimpleNamespace
from typing import List

from .SystemPool import SystemPool
from .VariationOperators import VariationOperators


logger = logging.getLogger(__name__)


class Target(object):
    """
    Target space is one of the most important concepts of global search algorithms. This space is formed by all the
    systems among which we perform our search and closely tied to our search. This class contains configuration of
    such space, parameters of what are we searching for, list of systems already studied in the search, current result
    of the search and tools to wisely create new systems for the search within this space.

    :ivar config:
        link to implementation of :class:`~USPEX.Common.Config.Config` interface.
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

    def __init__(self, type: str, **kwargs):
        """
        Initializes the class.

        :type type: str
        :param type: target type.
        :type kwargs: dict
        :param kwargs: parameters for initializing config.
        """
        targetTypes = self.knownTargetTypes[type]
        self.systemType = targetTypes.systemType
        self.config = kwargs['config']
        self.pool = SystemPool()
        self.utilities = {}
        for untilityType in targetTypes.sharedUtilities:
            name = untilityType.__name__[0].lower() + untilityType.__name__[1:]
            self.utilities[name] = untilityType(**kwargs[name]) if name in kwargs else untilityType()

        self.hybridizations = []
        for hybridizationType in targetTypes.variationOperators.hybridizationTypes:
            name = hybridizationType.__name__[0].lower() + hybridizationType.__name__[1:]
            params = kwargs[name] if name in kwargs else {}
            self.hybridizations.append(hybridizationType(self.systemType, self.config, self.pool, self.utilities,
                                                         **params))

        self.mutations = []
        for mutationType in targetTypes.variationOperators.mutationTypes:
            name = mutationType.__name__[0].lower() + mutationType.__name__[1:]
            params = kwargs[name] if name in kwargs else {}
            self.mutations.append(mutationType(self.systemType, self.config, self.pool, self.utilities, **params))

        self.creations = []
        for creationType in targetTypes.variationOperators.creationTypes:
            name = creationType.__name__[0].lower() + creationType.__name__[1:]
            params = kwargs[name] if name in kwargs else {}
            self.creations.append(creationType(self.systemType, self.config, self.pool, self.utilities, **params))

        self.variationOperators = self.hybridizations + self.mutations + self.creations

    @classmethod
    def registerTarget(cls, name: str, systemType: type, sharedUtilities: List[type], variationOperators: VariationOperators):
        """
        Register the target as known target.

        :type name: str
        :param name: target name.
        :type systemType: type
        :param systemType: system type.
        :type poolType: type
        :param poolType: pool type.
        :type variationOperators: :class:`VariationOperators`
        :param variationOperators: variation operators.
        """
        assert name not in cls.knownTargetTypes
        cls.knownTargetTypes[name] = SimpleNamespace(**{'systemType': systemType, 'sharedUtilities': sharedUtilities,
                                                        'variationOperators': variationOperators})
