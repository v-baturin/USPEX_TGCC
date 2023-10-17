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

from .SystemPool import SystemPool
from .PoolEntry import FlavourFactory
from .Target import Target, TargetType
from ..Selection.Antiseeds import Antiseeds
from USPEX.Expressions.Functions.BasicFunctions import BasicFunctions
from USPEX.Expressions.Functions.presets import applyPresetsRecursive


logger = logging.getLogger(__name__)


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
        self.pool = SystemPool(FlavourFactory(self.target.propertyExtensions))
        self.fingerprintUtility = getattr(self.target.utilities, fingerprintUtility)
        self.extraData = list(extraData)
        self.selectionConfig = selection
        self._createPopulation = self.knownSelectionTypes[selection['type']](self.pool, self.target,
                                                                             self.fingerprintUtility, **selection)

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

        self._mostDiverse = []

        self.best = set()
        self._isStable = False
        self._isGoalReached = False

    def __copy__(self):
        other = GlobalOptimizer.__new__(GlobalOptimizer)
        other.pool = copy(self.pool)
        other.target = copy(self.target)
        other.selectionConfig = self.selectionConfig
        other.createPopulation = copy(self.createPopulation)
        other.optType = self.optType
        other.stopFitness = self.stopFitness
        other.stopSystems = self.stopSystems
        other.best = self.best
        other._isStable = self._isStable
        other._isGoalReached = self._isGoalReached
        return other

    def createPopulation(self):
        if self.antiseeds.legacy:
            self.antiseeds.payPenalties(self.pool.generations[-1]['allSystems'],
                                        self.pool.uniqueSystems, self.fingerprintUtility)
        population = list(self.pool.uniqueSystems) if self.globalParentsPool else self.pool.generations[-1]['allSystems']
        offsprings = []
        self._createPopulation(population, offsprings)
        return offsprings

    async def update(self, population: list):
        """
        Updates state of optimized structures.

        :type population: list
        :param population: list of systems which allows to update our knowledge about target space.
        """
        goodSystems = []
        for system in population:
            for suffix in self.goodSystemsSuffixes:
                if system[f'.isBad.{suffix}']:
                    break
            else:
                self.pool.goodSystemIDs.append(system.ID)
                goodSystems.append(system)
        self.ExpressionEvaluator.calculate(self.optType, self.pool.goodSystems, self.extensions)
        self.ExpressionEvaluator.calculate(self._createPopulation.optType, self.pool.goodSystems, self.extensions)
        population = goodSystems
        assert population, 'All systems in population failed relaxation.'
        self._markDuplicates(population)
        logger.debug('Updating target: list of unique systems.')
        IDs = set(system['ID'] for system in population)
        newIDs = []
        newGeneration = {'allSystems': [], 'newSystems': []}
        for system in population:
            original = self.pool.allSystems[self._getOriginalID(system['ID'])]
            if original['ID'] not in newIDs:
                newGeneration['allSystems'].append(original)
                newIDs.append(original['ID'])
                if set(original.duplicates) <= IDs:
                    logger.debug(f'add new system {system["ID"]} to list of unique systems')
                    newGeneration['newSystems'].append(original)
        self.pool.generations.append(newGeneration)
        best = set(system['ID'] for system in self.pool.fronts(self.pool.uniqueSystems, self.optType)[0])
        if best == self.best:
            self._isStable = True
        else:
            self._isStable = False
            self.best = best
        self.pool.generations[-1]['bestSystems'] = self.best
        if self.stopFitness is not None:
            for ID in self.best:
                if round(self.pool.allSystems[self._getOriginalID(ID)][self.optType], ndigits=3)\
                        <= round(self.stopFitness, ndigits=3):
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

    def _markDuplicates(self, population: list):
        """
        Method for cleaning duplicates.

        :type population: list of :class:`~USPEX.Common.System.System` descendants
        :param population: list of systems which allows to update our knowledge about target space.
        """
        logger.info('Looking for duplicates.')
        for system in population:
            for i, ref_system in enumerate(self.pool.uniqueSystems):
                if self.fingerprintUtility.equal(system, ref_system) and system['ID'] != ref_system['ID']:
                    logger.info(f"system {system['ID']} coincides with system {ref_system['ID']} found earlier")
                    if system[applyPresetsRecursive(self.optType)] < ref_system[applyPresetsRecursive(self.optType)]:
                        self.fingerprintUtility.clean(ref_system)
                        ref_system.setProperty('originalID', system['ID'])
                        system.duplicates = ref_system.duplicates
                        for ID in system.duplicates:
                            self.pool.allSystems[ID].originalID = system['ID']
                        if ref_system['ID'] not in system.duplicates:
                            system.duplicates.append(ref_system['ID'])
                    else:
                        self.fingerprintUtility.clean(system)
                        system.setProperty('originalID', ref_system['ID'])
                        if system['ID'] not in ref_system.duplicates:
                            ref_system.duplicates.append(system['ID'])
                    break

    def _getOriginalID(self, ID):
        """
        If system is duplicate return ID of original system otherwise return input ID.

        :param ID: ID of some system from this pool.

        :return: ID of original system.
        """
        system = self.pool.allSystems[ID]
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
