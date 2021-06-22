"""
USPEX.Common.GlobalOptimizer
============================

Class implementing global optimizer

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
from copy import copy
from typing import List

from .SystemPool import SystemPool
from .Target import Target, TargetType

logger = logging.getLogger(__name__)


class GlobalOptimizer(object):
    """
    Main purpose of this class is to generate new structures

    :ivar best:
        list of currently best known systems.

    """

    Fitness = None
    knownSelectionTypes = {}
    knownTargetTypes = {}

    @classmethod
    def setFitnessType(cls, FitnessType: type):
        cls.Fitness = FitnessType

    @classmethod
    def registerSelection(cls, selectionType: type):
        assert selectionType.__name__ not in cls.knownSelectionTypes
        cls.knownSelectionTypes[selectionType.__name__] = selectionType

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

    def __init__(self, target: dict, selection: dict, optType, fingerprintUtility, stopFitness=None, stopSystems=None, **kwargs):
        """
        Initializes the class.

        :type target: dict{type, params}
        :param target: name of target system and its parameters; obligatory
        :type selection: dict{type, params}
        :param selection: name of selection to launch and its parameters; obligatory
        """

        self.pool = SystemPool()
        self.target = Target(self.knownTargetTypes[target['type']], **target)
        self.fingerprintUtility = getattr(self.target.utilities, fingerprintUtility)
        self.fitness = self.Fitness(self.pool, self.target.utilities, self.fingerprintUtility)
        self.selectionConfig = selection
        self.createPopulation = self.knownSelectionTypes[selection['type']](self.pool, self.target, self.fitness,
                                                                           self.fingerprintUtility ,**selection)

        self.optType = optType
        self.stopFitness = stopFitness
        if stopSystems is not None and self.target.seeds is not None:
            Seeds = type(self.target.seeds)
            seeds = Seeds(self.target.utilities, generations = [0], seedsFolders=[stopSystems])
            self.stopSystems = seeds()
        else:
            self.stopSystems = None

        self.best = set()
        self._isStable = False
        self._isGoalReached = False

    def __copy__(self):
        other = GlobalOptimizer.__new__(GlobalOptimizer)
        other.pool = copy(self.pool)
        other.target = copy(self.target)
        other.fitness = copy(self.fitness)
        other.fitness.pool = other.pool
        other.fitness.utilities = other.target.utilities
        other.selectionConfig = self.selectionConfig
        other.createPopulation = copy(self.createPopulation)
        other.optType = self.optType
        other.stopFitness = self.stopFitness
        other.stopSystems = self.stopSystems
        other.best = self.best
        other._isStable = self._isStable
        other._isGoalReached = self._isGoalReached
        return other

    def update(self, population: list):
        """
        Updates state of optimized structures.

        :type population: list
        :param population: list of systems which allows to update our knowledge about target space.
        """
        self._cleanDuplicates(population)
        self.pool.update(population)
        allFitnesses = self.fitness.getAllFitnesses(self.optType)
        for VO in self.target.variationOperators:
            if hasattr(VO, 'tune'):
                VO.tune(population, allFitnesses)
        best = set(system['ID'] for system in self.fitness.sort(list(self.pool.uniqueSystems), allFitnesses)[0])
        if best == self.best:
            self._isStable = True
        else:
            self._isStable = False
            self.best = best
        if self.stopFitness is not None:
            for ID in self.best:
                if round(self.fitness.getFitnessByID(self.optType, ID), ndigits=3) <= round(self.stopFitness, ndigits=3):
                    self._isGoalReached = True
                    break
        if self.stopSystems is not None and not self._isGoalReached:
            stopSystems = list(self.stopSystems)
            for system in self.pool.uniqueSystems:
                for i, stopSystem in enumerate(stopSystems):
                    if self.fingerprintUtility.equal(system, stopSystem):
                        del stopSystems[i]
                        break
                if not stopSystems:
                    break
            self._isGoalReached = not stopSystems

    def _cleanDuplicates(self, population: list):
        """
        Method for cleaning duplicates.

        :type population: list of :class:`~USPEX.Common.System.System` descendants
        :param population: list of systems which allows to update our knowledge about target space.
        """
        logger.info('Looking for duplicates.')
        cleanedPopulation = []
        for system in population:
            for ref_system in list(self.pool.uniqueSystems) + cleanedPopulation:
                if self.fingerprintUtility.equal(system, ref_system):
                    logger.info(f"system {system['ID']} coincides with system {ref_system['ID']} found earlier")
                    self.fingerprintUtility.clean(system)
                    system['originalID'] = ref_system['ID']
                    system = ref_system
                    break

            if not system['isBad']:
                cleanedPopulation.append(system)
        assert cleanedPopulation, 'All systems in population failed relaxation.'
        population[:] = cleanedPopulation


    @property
    def isStable(self):
        return self._isStable

    @property
    def isGoalReached(self):
        return self._isGoalReached
