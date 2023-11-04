"""
USPEX.Common.GlobalOptimizer
============================

Class implementing global optimizer

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
from copy import copy
from typing import List
from itertools import chain

from .PoolEntry import FlavourFactory, Pool
from .Target import Target, TargetType
from ..Expressions.Functions.BasicFunctions import BasicFunctions
from ..Expressions.Functions.presets import applyPresetsRecursive
from ..Expressions.Antiseeds import Antiseeds


logger = logging.getLogger(__name__)


class Generation:
    population = None
    goodPopulation = None
    uniquePopulation = None
    goodSystems = None
    uniqueSystems = None


class GlobalOptimizer(object):
    """
    Main purpose of this class is to generate new structures

    :ivar best:
        list of currently best known systems.

    """

    ExpressionEvaluator = None
    knownSelectionTypes = {}
    knownTargetTypes = {}

    @classmethod
    def setExpressionEvaluatorType(cls, ExpressionEvaluatorType: type):
        cls.ExpressionEvaluator = ExpressionEvaluatorType

    @classmethod
    def registerSelection(cls, selectionType: type):
        assert selectionType.__name__ not in cls.knownSelectionTypes, f'{selectionType.__name__} is not set as known Selection type'
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
        assert name not in cls.knownTargetTypes, f'{name} is not registered as known Target'
        cls.knownTargetTypes[name] = TargetType(utilities=utilities, hybridizations=hybridizations,
                                                mutations=mutations, creations=creations, seeds=seeds)

    def __init__(self, target: dict, selection: dict, optType, fingerprintUtility, stopFitness=None, stopSystems=None,
                 extraData=(), antiseeds: dict = None, **kwargs):
        """
        Initializes the class.

        :type target: dict{type, params}
        :param target: name of target system and its parameters; obligatory
        :type selection: dict{type, params}
        :param selection: name of selection to launch and its parameters; obligatory
        """

        self.extensions = {'basic': BasicFunctions()}
        self.target = Target(self.knownTargetTypes[target['type']], **target)
        self.extensions.update(**self.target.expressionExtensions)
        self.flavourFactory = FlavourFactory(self.target.propertyExtensions)
        self.fingerprintUtility = getattr(self.target.utilities, fingerprintUtility)
        self.extraData = list(extraData)
        antiseeds = {} if antiseeds is None else antiseeds
        self.antiseeds = Antiseeds(self.fingerprintUtility, **antiseeds)
        self.flavourFactory.extensions['antiseeds'] = self.antiseeds

        self._createPopulation = self.knownSelectionTypes[selection['type']](self.target, self.fingerprintUtility,
                                                                             **selection)

        self.optType = optType
        self.stopFitness = stopFitness
        if stopSystems is not None and self.target.seeds is not None:
            seeds = type(self.target.seeds)(self.target.utilities, generations=[0], seedsFolders=[stopSystems])
            self.stopSystems = seeds()
        else:
            self.stopSystems = None

        self.goodSystemsSuffixes = set(
            prop.split('.')[-1] for prop in _extract(self.optType) + _extract(self._createPopulation.optType)
        )

        self.allSystems = Pool.createPool(self.flavourFactory)
        self.generations: list[Generation] = []

        self.best = set()
        self.bestHistory = []
        self._isStable = False
        self._isGoalReached = False

    def createPopulation(self):
        if self.generations:
            generation = self.generations[-1]
            self.antiseeds.payPenalties(generation.uniquePopulation, generation.uniqueSystems)
            population = generation.uniqueSystems if self._createPopulation.globalParentsPool else generation.uniquePopulation
            optType = generation.goodSystems.createExpression(self._createPopulation.optType)
        else:
            population = None
            optType = None
        offsprings = Pool.createPool(self.flavourFactory)
        self._createPopulation(population, offsprings, optType)
        for ID in offsprings.getIDs():
            self.allSystems.addEntry(offsprings.getEntry(ID))
        return offsprings

    async def update(self, population):
        """
        Updates state of optimized structures.

        :param population: list of systems which allows to update our knowledge about target space.
        """
        generation = Generation()
        generation.population = population
        generation.goodPopulation = Pool.createPool(self.flavourFactory)
        if self.generations:
            generation.goodSystems = copy(self.generations[-1].goodSystems)
        else:
            generation.goodSystems = Pool.createPool(self.flavourFactory)
        for ID in population.getIDs():
            system = population.getEntry(ID)
            for suffix in self.goodSystemsSuffixes:
                if system[f'.isBad.{suffix}']:
                    break
            else:
                generation.goodSystems.addEntry(system)
                generation.goodPopulation.addEntry(system)
        self.ExpressionEvaluator.calculate(self.optType, generation.goodSystems, self.extensions)
        self.ExpressionEvaluator.calculate(self._createPopulation.optType, generation.goodSystems, self.extensions)
        assert generation.goodPopulation.getIDs(), 'All systems in population failed relaxation.'
        optType = self.optType if isinstance(self.optType, str) else generation.goodSystems.createExpression(self.optType)
        self._markDuplicates(generation.goodPopulation, optType)
        generation.uniqueSystems = Pool.createPool(self.flavourFactory)
        for ID in generation.goodSystems.getIDs():
            entry = generation.goodSystems.getEntry(ID)
            if entry.originalID is None:
                generation.uniqueSystems.addEntry(entry)
        logger.debug('Updating target: list of unique systems.')
        newIDs = []
        generation.uniquePopulation = Pool.createPool(self.flavourFactory)
        for ID in generation.goodPopulation.getIDs():
            system = generation.goodSystems.getEntry(ID)
            original = self.allSystems.getEntry(self._getOriginalID(system['ID']))
            if original['ID'] not in newIDs:
                generation.uniquePopulation.addEntry(original)
                newIDs.append(original['ID'])
        self.generations.append(generation)
        best = set(system['ID'] for system in generation.uniqueSystems.fronts(optType)[0])
        if best == self.best:
            self._isStable = True
        else:
            self._isStable = False
            self.best = best
        self.bestHistory.append(self.best)
        if self.stopFitness is not None:
            for ID in self.best:
                if round(self.allSystems[self._getOriginalID(ID)][optType], ndigits=3)\
                        <= round(self.stopFitness, ndigits=3):
                    self._isGoalReached = True
                    break
        if self.stopSystems is not None and not self._isGoalReached:
            stopSystems = list(self.stopSystems)
            for system in generation.uniqueSystems:
                for i, stopSystem in enumerate(stopSystems):
                    if self.fingerprintUtility.equal(system, stopSystem):
                        del stopSystems[i]
                        break
                if not stopSystems:
                    break
            self._isGoalReached = not stopSystems

    def _markDuplicates(self, population, optType):
        """
        Method for cleaning duplicates.

        :type population: list of :class:`~USPEX.Common.System.System` descendants
        :param population: list of systems which allows to update our knowledge about target space.
        """
        logger.info('Looking for duplicates.')
        if self.generations:
            uniqueSystems = self.generations[-1].uniqueSystems.getIDs()
        else:
            uniqueSystems = []
        for system_ID in population.getIDs():
            system = population.getEntry(system_ID)
            for i, ref_system_ID in enumerate(uniqueSystems):
                ref_system = self.allSystems.getEntry(ref_system_ID)
                if self.fingerprintUtility.equal(system, ref_system) and system['ID'] != ref_system['ID']:
                    logger.info(f"system {system['ID']} coincides with system {ref_system['ID']} found earlier")
                    if system[applyPresetsRecursive(optType)] < ref_system[applyPresetsRecursive(optType)]:
                        self.fingerprintUtility.clean(ref_system)
                        ref_system.setProperty('originalID', system['ID'])
                        system.duplicates = ref_system.duplicates
                        for ID in system.duplicates:
                            self.allSystems.getEntry(ID).originalID = system['ID']
                        if ref_system['ID'] not in system.duplicates:
                            system.duplicates.append(ref_system['ID'])
                        uniqueSystems[i] = system.ID
                    else:
                        self.fingerprintUtility.clean(system)
                        system.setProperty('originalID', ref_system['ID'])
                        if system['ID'] not in ref_system.duplicates:
                            ref_system.duplicates.append(system['ID'])
                    break
            else:
                uniqueSystems.append(system.ID)

    def _getOriginalID(self, ID):
        """
        If system is duplicate return ID of original system otherwise return input ID.

        :param ID: ID of some system from this pool.

        :return: ID of original system.
        """
        system = self.allSystems.getEntry(ID)
        return system.originalID if system.originalID is not None else ID

    @property
    def isStable(self):
        return self._isStable

    @property
    def isGoalReached(self):
        return self._isGoalReached


def _extract(expression):
    if isinstance(expression, str):
        return [expression]
    if isinstance(expression, tuple):
        func, *arguments = expression
        return list(set(chain(*[_extract(arg) for arg in arguments])))
    else:
        return []
