"""
USPEX.Common.GlobalOptimizer
============================

Class implementing global optimizer

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
from copy import copy
from typing import List, Tuple

from .Target import Target

logger = logging.getLogger(__name__)


class GlobalOptimizer(object):
    """
    Main purpose of this class is to generate new structures

    :ivar best:
        list of currently best known systems.

    """

    Fitness = None
    knownSelectionTypes = {}

    def __init__(self, target: dict, selection: dict, fitness, fingerprintUtility, stopFitness=None, **kwargs):
        """
        Initializes the class.

        :type target: dict{type, params}
        :param target: name of target system and its parameters; obligatory
        :type selection: dict{type, params}
        :param selection: name of selection to launch and its parameters; obligatory
        """

        self.target = Target(**target)
        self.fingerprintUtility = getattr(self.target.utilities, fingerprintUtility)

        assert self.Fitness is not None
        self.fitness = self.Fitness(self.target.pool, self.target.utilities, self.fingerprintUtility)
        self.fitnessConvergence = fitness
        self.best = set()
        self._isStable = False
        self.stopFitness = stopFitness
        self._isGoalReached = False

        self.selectionConfig = selection
        self.createPopulation = self.knownSelectionTypes[selection['type']](self.fingerprintUtility ,**selection)

        # List of structure recieved from update on this particular step
        self.population = None
        # List of new found structure on this particular step
        self.newStructures = None

    def __copy__(self):
        other = GlobalOptimizer.__new__(GlobalOptimizer)
        other.target = copy(self.target)
        other.fitness = copy(self.fitness)
        other.fitness.pool = other.target.pool
        other.fitness.utilities = other.target.utilities
        other.fitnessConvergence = self.fitnessConvergence
        other.best = copy(self.best)
        other._isStable = self._isStable
        other.stopFitness = self.stopFitness
        other._isGoalReached = self._isGoalReached
        other.selectionConfig = self.selectionConfig
        other.createPopulation = self.createPopulation
        other.population = copy(self.population)
        other.newStructures = copy(self.newStructures)
        return other

    def run(self):
        """
        Here we generate new set of structures.

        :rtype: list
        :return: list of structures.
        """
        return self.createPopulation(self.target, self.fitness, self.population, self.newStructures)

    def update(self, population: list):
        """
        Updates state of optimized structures.

        :type population: list
        :param population: list of systems which allows to update our knowledge about target space.
        """
        self.cleanDuplicates(population)
        self.population = population
        self.newStructures = self.target.pool.newFoundSystems(population)
        self.target.pool.update(self.newStructures)
        best = set(system['ID'] for system in self.fitness.sort(self.fitnessConvergence, list(self.target.pool.uniqueSystems))[0])
        if best == self.best:
            self._isStable = True
        else:
            self._isStable = False
            self.best = best
        if self.stopFitness is not None:
            for ID in list(self.best):
                value = self.fitness.getFitnessByID(self.fitnessConvergence, ID)
                if value is None:
                    try:
                        value = self.target.pool.allSystems[ID][self.fitnessConvergence]
                    except:
                        pass
                if round(value, ndigits=3) <= round(self.stopFitness, ndigits=3):
                    self._isGoalReached = True

    def cleanDuplicates(self, population: list):
        """
        Method for cleaning duplicates.

        :type population: list of :class:`~USPEX.Common.System.System` descendants
        :param population: list of systems which allows to update our knowledge about target space.
        """
        logger.info('Looking for duplicates.')
        cleanedPopulation = []
        for system in population:
            for ref_system in list(self.target.pool.uniqueSystems) + cleanedPopulation:
                if self.fingerprintUtility.equal(system, ref_system):
                    logger.info(f"system {system['ID']} coincides with system {ref_system['ID']} found earlier")
                    self.fingerprintUtility.clean(system)
                    system = ref_system
                    break

            if not system['isBad']:
                cleanedPopulation.append(system)
        population[:] = cleanedPopulation


    @property
    def isStable(self):
        return self._isStable

    @property
    def isGoalReached(self):
        return self._isGoalReached

    @classmethod
    def setFitnessType(cls, FitnessType: type):
        cls.Fitness = FitnessType

    @classmethod
    def registerSelection(cls, selectionType: type):
        assert selectionType.__name__ not in cls.knownSelectionTypes
        cls.knownSelectionTypes[selectionType.__name__] = selectionType
