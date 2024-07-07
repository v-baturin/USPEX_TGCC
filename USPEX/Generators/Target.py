"""
USPEX.Common.Target
===================

Class describing target space

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
"""

import logging
from types import SimpleNamespace

from ..DataModel.Flavour import FlavourFactory
from ..DataModel.Pool import Pool
from ..DataModel.Generations import Generations


logger = logging.getLogger(__name__)


class TargetType:
    def __init__(self, utilities: list[type], hybridizations: list[type], mutations: list[type], creations: list[type],
                 defaultMetric: str, seeds: type = None):
        self.utilities = utilities
        self.hybridizations = hybridizations
        self.mutations = mutations
        self.creations = creations
        self.seeds = seeds
        self.defaultMetric = defaultMetric


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

    knownTargetTypes: dict[str, TargetType] = {}

    @classmethod
    def registerTarget(cls, name: str, utilities: list[type], hybridizations: list[type], mutations: list[type],
                       creations: list[type], defaultMetric: str, seeds: type = None):
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
        assert name not in cls.knownTargetTypes, f'{name} is not registered as known Target'
        cls.knownTargetTypes[name] = TargetType(utilities=utilities, hybridizations=hybridizations,
                                                mutations=mutations, creations=creations, seeds=seeds,
                                                defaultMetric=defaultMetric)

    def __init__(self, type: str, defaultSuffix: str, **kwargs):
        """
        Initializes the class.

        :type type:
        :param type: target types.
        :type kwargs: dict
        :param kwargs: parameters for initializing config.
        """
        targetTypes = self.knownTargetTypes[type]
        self.defaultSuffix = defaultSuffix
        self.name = type
        utilities = {}
        self.expressionExtensions = {}
        self.propertyExtensions = {}
        failedUtilities = []
        for utilityType in targetTypes.utilities:
            name = utilityType.__name__[0].lower() + utilityType.__name__[1:]
            try:
                utility = utilityType(**kwargs[name]) if name in kwargs else utilityType()
                utilities[name] = utility
                if hasattr(utilityType, 'expressionExtension'):
                    self.expressionExtensions[name] = (utility, utility.expressionExtension.expressionTable)
                if hasattr(utility, 'propertyExtension'):
                    self.propertyExtensions[name] = (utility, utility.propertyExtension.propertyTable)
            except TypeError as e:
                logger.debug(e)
                failedUtilities.append(utilityType.__name__)
            except Exception as e:
                logger.error(e, exc_info=True)
        logger.info(f'Following utilities was not initialized: {failedUtilities}.')
        self.utilities = SimpleNamespace(**utilities)

        self.hybridizations = []
        for hybridizationType in targetTypes.hybridizations:
            name = hybridizationType.__name__[0].lower() + hybridizationType.__name__[1:]
            params = kwargs[name] if name in kwargs else {}
            if 'suffix' not in params:
                params['suffix'] = defaultSuffix
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
            if 'suffix' not in params:
                params['suffix'] = defaultSuffix
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
        self.metric = getattr(self.utilities, targetTypes.defaultMetric)
        self.flavourFactory = FlavourFactory(self.propertyExtensions, self.metric)

    def createPool(self) -> Pool:
        return Pool.newPool(self.flavourFactory, self.expressionExtensions, self.metric)

    def createGenerations(self) -> Generations:
        return Generations(self.flavourFactory, self.expressionExtensions, self.metric)
