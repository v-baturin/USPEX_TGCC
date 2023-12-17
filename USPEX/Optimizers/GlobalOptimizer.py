"""
USPEX.Common.GlobalOptimizer
============================

Class implementing global optimizer

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
from copy import copy

import numpy as np

from ..Expressions.Functions.presets import applyPresetsRecursive


logger = logging.getLogger(__name__)


class Generation:
    population = None
    goodPopulation = None
    uniquePopulation = None
    goodSystems = None
    uniqueSystems = None
    best = None


class GlobalOptimizer(object):
    """
    Main purpose of this class is to generate new structures

    :ivar best:
        list of currently best known systems.

    """

    knownSelectionTypes = {}
    Target = None

    @classmethod
    def registerSelection(cls, selectionType: type):
        assert selectionType.__name__ not in cls.knownSelectionTypes, f'{selectionType.__name__} is not set as known Selection type'
        cls.knownSelectionTypes[selectionType.__name__] = selectionType

    @classmethod
    def setTarget(cls, targetType):
        cls.Target = targetType

    def __init__(self, target: dict, selection: dict, optType, goodSystemsSuffixes, stopValue=None, stopSystems=None,
                 **kwargs):
        """
        Initializes the class.

        :type target: dict{type, params}
        :param target: name of target system and its parameters; obligatory
        :type selection: dict{type, params}
        :param selection: name of selection to launch and its parameters; obligatory
        """

        self.target = self.Target(**target)
        self._createPopulation = self.knownSelectionTypes[selection['type']](self.target, **selection)

        self.optType = applyPresetsRecursive(optType)
        self.goodSystemsSuffixes = goodSystemsSuffixes
        self.stopValue = stopValue
        self.stopSystems = stopSystems

    def createPopulation(self, generation):
        return self._createPopulation(generation)

    async def update(self, population, parentsGeneration):
        """
        Updates state of optimized structures.

        :param population: list of systems which allows to update our knowledge about target space.
        """
        generation = Generation()
        generation.population = population
        generation.goodPopulation = population.createPool()
        if parentsGeneration is not None:
            generation.goodSystems = copy(parentsGeneration.goodSystems)
            oldBest = parentsGeneration.best
        else:
            generation.goodSystems = population.createPool()
            oldBest = None
        for ID in population.getIDs():
            system = population.getEntry(ID)
            for suffix in self.goodSystemsSuffixes:
                if suffix not in system.flavours or system[f'.isBad.{suffix}']:
                    break
            else:
                generation.goodSystems.addEntry(system)
                generation.goodPopulation.addEntry(system)
        generation.goodSystems.evaluate(self.optType)
        assert generation.goodPopulation.getIDs(), 'All systems in population failed relaxation.'
        optType = generation.goodSystems.createExpression(self.optType)
        generation.uniqueSystems = self._markDuplicates(generation.goodPopulation, generation.goodSystems,
                                                        parentsGeneration, optType)
        logger.debug('Updating target: list of unique systems.')
        generation.uniquePopulation = population.createPool()
        for ID in generation.goodPopulation.getIDs():
            system = generation.goodPopulation.getEntry(ID)
            try:
                system = generation.goodPopulation.getEntry(system.getProperty('originalID'))
            except KeyError:
                pass
            if system.ID not in generation.uniquePopulation.getIDs():
                generation.uniquePopulation.addEntry(system)
        best = set(system.ID for system in generation.uniqueSystems.fronts(optType)[0])
        if best == oldBest:
            isStable = True
        else:
            isStable = False
        generation.best = best
        isGoalReached = False
        if self.stopValue is not None:
            for ID in best:
                value = generation.uniqueSystems.getEntry(ID)[optType]
                if value < self.stopValue or np.isclose(value, self.stopValue, atol=5.e-4):
                    isGoalReached = True
                    break
        if self.stopSystems is not None and not isGoalReached:
            stopSystems = list(self.stopSystems)
            for system in generation.uniqueSystems:
                for i, stopSystem in enumerate(stopSystems):
                    if self.target.metric.equal(system, stopSystem):
                        del stopSystems[i]
                        break
                if not stopSystems:
                    break
            isGoalReached = not stopSystems
        return generation, isStable, isGoalReached

    def _markDuplicates(self, population, goodSystems, parentsGeneration, optType):
        """
        Method for cleaning duplicates.

        :type population: list of :class:`~USPEX.Common.System.System` descendants
        :param population: list of systems which allows to update our knowledge about target space.
        """
        logger.info('Looking for duplicates.')
        if parentsGeneration is not None:
            uniqueSystems = parentsGeneration.uniqueSystems.getIDs()
        else:
            uniqueSystems = []
        for system_ID in population.getIDs():
            system = population.getEntry(system_ID)
            for i, ref_system_ID in enumerate(uniqueSystems):
                ref_system = goodSystems.getEntry(ref_system_ID)
                if self.target.metric.equal(system, ref_system) and system.ID != ref_system.ID:
                    logger.info(f"system {system.ID} coincides with system {ref_system.ID} found earlier")
                    try:
                        duplicates = ref_system.getProperty('duplicates')
                    except KeyError:
                        duplicates = []
                    if system[optType] < ref_system[optType]:
                        ref_system.setProperty('originalID', system.ID)
                        for ID in duplicates:
                            goodSystems.getEntry(ID).setProperty('originalID', system.ID)
                        if ref_system.ID not in duplicates:
                            duplicates.append(ref_system.ID)
                        system.setProperty('duplicates', duplicates)
                        ref_system.delProperty('duplicates')
                        uniqueSystems[i] = system.ID
                    else:
                        system.setProperty('originalID', ref_system.ID)
                        if system.ID not in duplicates:
                            duplicates.append(system.ID)
                        ref_system.setProperty('duplicates', duplicates)
                    break
            else:
                uniqueSystems.append(system.ID)
        uniqueSystemsPool = population.createPool()
        for ID in uniqueSystems:
            uniqueSystemsPool.addEntry(goodSystems.getEntry(ID))
        return uniqueSystemsPool
