"""
USPEX.Common.Target
===================

Class describing target space

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
"""

import logging

from types import SimpleNamespace

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
        targetDef = SimpleNamespace(**self.knownTargetTypes[type])
        self.config = targetDef.configType(**kwargs)
        self.pool = targetDef.poolType(self.config)
        
        self.hybridizations = []
        for hybridizationType in targetDef.variationOperators.hybridizationTypes:
            if hybridizationType.__name__ in kwargs:
                self.hybridizations.append(hybridizationType(self.config, self.pool,
                                                             **kwargs[hybridizationType.__name__]))

        self.mutations = []
        for mutationType in targetDef.variationOperators.mutationTypes:
            if mutationType.__name__ in kwargs:
                self.mutations.append(mutationType(self.config, self.pool, **kwargs[mutationType.__name__]))

        self.creations = []
        for creationType in targetDef.variationOperators.creationTypes:
            if creationType.__name__ in kwargs:
                self.creations.append(creationType(self.config, self.pool, **kwargs[creationType.__name__]))

        self.variationOperators = self.hybridizations + self.mutations + self.creations

    @classmethod
    def registerTarget(cls, name: str, configType: type, poolType: type, variationOperators: VariationOperators):
        """
        Register the target as known target.

        :type name: str
        :param name: target name.
        :type configType: type
        :param configType: config type.
        :type poolType: type
        :param poolType: pool type.
        :type variationOperators: :class:`VariationOperators`
        :param variationOperators: variation operators.
        """
        assert name not in cls.knownTargetTypes
        cls.knownTargetTypes[name] = {'configType': configType, 'poolType': poolType,
                                      'variationOperators': variationOperators}
