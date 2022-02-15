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


logger = logging.getLogger(__name__)


class TargetType:
    def __init__(self, utilities: List[type], hybridizations: List[type], mutations: List[type], creations: List[type],
                 seeds: type = None):
        self.utilities = utilities
        self.hybridizations = hybridizations
        self.mutations = mutations
        self.creations = creations
        self.seeds = seeds


class Target(object):
    """
    Class containing Utilities and variation operators for work with specific target space such as atomic structures,
    phase transition pathways, other USPEX calculations.

    :ivar varOperators:
        list of all variation operators for this target space. Each element is an implementation of
        :class:`~USPEX.Common.VarOperator.VarOperator` interface. For default implementation this list is empty.
    :ivar hybridizations:
        list of variation operators with two parents for this target space.
    :ivar mutations:
        list of variation operators with one parent for this target space.
    :ivar creations:
        list of variation operators with no parents for this target space.
    :ivar utilities:
        list of utilities.
    """

    def __init__(self, targetTypes : TargetType, **kwargs):
        """
        Initializes the class.

        :type type:
        :param type: target types.
        :type kwargs: dict
        :param kwargs: parameters for initializing config.
        """
        self.name = kwargs['type']
        utilities = {}
        failedUtilities = []
        self.constraintsType = None
        for untilityType in targetTypes.utilities:
            name = untilityType.__name__[0].lower() + untilityType.__name__[1:]
            if name == 'constraints':
                self.constraintsType = untilityType
            else:
                try:
                    utilities[name] = untilityType(**kwargs[name]) if name in kwargs else untilityType()
                except TypeError as e:
                    logger.debug(e)
                    failedUtilities.append(untilityType.__name__)
                except Exception as e:
                    logger.error(e, exc_info=True)
        logger.info(f'Following utilities was not initialized: {failedUtilities}.')
        self.utilities = SimpleNamespace(**utilities)
        assert self.constraintsType is not None
        self.constraints = self.constraintsType(self.utilities)

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
