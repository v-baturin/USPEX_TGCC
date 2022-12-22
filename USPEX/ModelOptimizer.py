"""
USPEX.Common.ModelOptimizer
===========================

Class implementing model optimizer

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
from copy import copy
from typing import List

from .Target import Target, TargetType

logger = logging.getLogger(__name__)


class External:

    executorType = None

    @classmethod
    def setExecutorType(cls, executorType):
        cls.executorType = executorType

    def __init__(self, interface):
        self.executor = self.executorType(**interface)
        self.isStable = False

    async def update(self, population):
        system = dict(population=population)
        await self.executor.run(system)
        self.isStable = system['isStable']


class ModelOptimizer(object):
    """
    Main purpose of this class is to generate new structures

    :ivar best:
        list of currently best known systems.

    """

    knownModelTypes = {}
    knownTargetTypes = {}

    @classmethod
    def registerModel(cls, modelType: type):
        assert modelType.__name__ not in cls.knownModelTypes
        cls.knownModelTypes[modelType.__name__] = modelType

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

    def __init__(self, target: dict, model: dict, popSize: int, initialPopSize: int, fractions: dict, **kwargs):
        """
        Initializes the class.

        :type target: dict{type, params}
        :param target: name of target system and its parameters; obligatory
        :type selection: dict{type, params}
        :param selection: name of selection to launch and its parameters; obligatory
        """

        self.target = Target(self.knownTargetTypes[target['type']], **target)
        self.model = self.knownModelTypes[model['type']](**model)
        self.popSize = popSize
        self.initialPopSize = initialPopSize
        self.fractions = fractions
        self.update = self.model.update
        self._isGoalReached = False
        self.firstCall = True

    def __copy__(self):
        other = ModelOptimizer.__new__(ModelOptimizer)
        other.target = copy(self.target)
        other._isGoalReached = self._isGoalReached
        return other

    def createPopulation(self):
        popSize = self.initialPopSize if self.firstCall else self.popSize
        self.firstCall = False
        population = []
        for creation in self.target.creations:
            howCome = type(creation).__name__
            howMany = popSize * self.fractions[howCome]
            howMany = 0 if howMany < 0 else howMany
            if hasattr(creation, 'prepare'):
                creation.prepare()
            for i in range(2 * howMany):
                if howMany <= 0:
                    break
                try:
                    offsprings = creation()
                    for offspring in offsprings:
                        offspring['howCome'] = howCome
                        logger.info(f"System {offspring['ID']} successfully created by {offspring['howCome']} operator.")
                    population.extend(offsprings)
                    howMany -= len(offsprings)
                except RuntimeError as e:
                    logger.debug(e, exc_info=True)
                except Exception as e:
                    logger.error(e, exc_info=True)
            if hasattr(creation, 'standby'):
                creation.standby()

    @property
    def isStable(self):
        return self.model.isStable

    @property
    def isGoalReached(self):
        return self._isGoalReached
