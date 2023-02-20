"""
USPEX.Common.Fitness
====================

Data type representing rules of how we determine which systems are better.

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
import numpy as np
from copy import copy
from collections.abc import Mapping
from sklearn.decomposition import PCA

from .ConvexHull import ConvexHull
from .paretoRanking import paretoRanking
from ..Presets import presetFitness


logger = logging.getLogger(__name__)


class Fitness:

    def __init__(self, uniqueSystems, utilities, extraData=None):
        self.uniqueSystems = uniqueSystems
        self.extraData = extraData if extraData is not None else []
        self.utilities = utilities
        self._storedFitnesses = {}

    @staticmethod
    def calculate(uniqueSystems, optType, utilities, extraData=None):
        fitness = Fitness(uniqueSystems, utilities, extraData)
        fitness.calcFitness(optType)
        return fitness

    def getAllFitnesses(self, optType):
        optType = Fitness.applyPresets(optType)
        if optType not in self._storedFitnesses:
            self.calcFitness(optType)
        return dict(zip([s['ID'] for s in self.uniqueSystems], self._storedFitnesses[optType]))

    def getFitnessByID(self, optType, ID: int):
        fitness = self.getAllFitnesses(optType)
        return fitness[ID] if ID in fitness else None

    def getFitnessDirect(self, optType, system: dict):
        if optType in system:
            value = system[optType]
        else:
            utility, suffix, *extra = optType.split('.')
            assert not extra, f"Too complex property {optType}."
            value = getattr(getattr(self.utilities, utility), suffix)(system)
        return value

    def calcFitness(self, optType):
        optType = Fitness.applyPresets(optType)

        if len(self.uniqueSystems) == 0:
            self._storedFitnesses[optType] = np.empty(0)
        elif optType not in self._storedFitnesses:
            if isinstance(optType, tuple):
                funcName, *funcParams = optType
                if not isinstance(funcName, str):
                    raise RuntimeError(f'Incorrect type {type(funcName)} of function {funcName}.')
                else:
                    funcName = funcName.split('.')
                    arguments = [self.calcFitness(param) for param in funcParams]
                    size = min(len(arg) for arg in arguments if hasattr(arg, '__len__'))
                    for i, arg in enumerate(arguments):
                        if hasattr(arg, '__len__'):
                            arguments[i] = arg[:size]
                    if len(funcName) == 1:
                        funcName, = funcName
                        self._storedFitnesses[optType] = getattr(self, funcName)(*arguments)
                    elif len(funcName) == 2:
                        utility, funcName = funcName
                        self._storedFitnesses[optType] = getattr(getattr(self.utilities, utility), funcName)(*arguments)
                    else:
                        raise RuntimeError(f"Too complex fitness {'.'.join(optType)}.")
            elif isinstance(optType, str):
                value = [self.getFitnessDirect(optType, system) for system in self.uniqueSystems] + \
                        [system[optType] for system in self.extraData if optType in system]
                # unfortunately simple np.asarray spoils dictionaries
                if value and isinstance(value[0], Mapping):
                    valueArray = np.empty((len(value,)), dtype=type(value[0]))
                    for i, x in enumerate(value):
                        valueArray[i] = x
                else:
                    valueArray = np.asarray(value)
                self._storedFitnesses[optType] = valueArray
            else:
                # just a parameter. return it without doing anything.
                return optType
        return self._storedFitnesses[optType]

    @staticmethod
    def applyPresets(optType):
        optType_ref = optType
        if optType in presetFitness:
            optType = presetFitness[optType]
        elif isinstance(optType, tuple):
            funcName, *funcParams = optType
            for probeFitness in presetFitness.keys():
                if isinstance(probeFitness, tuple) and probeFitness[0] == funcName and len(probeFitness) == len(optType):
                    funcName, *templateParams = probeFitness
                    optType = presetFitness[probeFitness]
                    for param, templateParam in zip(funcParams, templateParams):
                        optType = Fitness._substituteParams(optType, templateParam, param)
                    break
        if optType != optType_ref:
            optType = Fitness.applyPresets(optType)
        return optType

    @staticmethod
    def _substituteParams(optType, templateParam, param):
        if optType == templateParam:
            optType = param
        elif isinstance(optType, tuple):
            funcName, *funcParams = optType
            optType = (funcName,)
            for funcParam in funcParams:
                optType += (Fitness._substituteParams(funcParam, templateParam, param),)
        return optType

    @staticmethod
    def min(values: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.min(values, axis=0),axis = 0).repeat(len(values), axis=0)

    @staticmethod
    def max(values: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.max(values, axis=0),axis = 0).repeat(len(values), axis=0)

    @staticmethod
    def mean(values: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.mean(values, axis=0),axis = 0).repeat(len(values), axis=0)

    @staticmethod
    def plus(values1: np.ndarray, values2: np.ndarray) -> np.ndarray:
        return values1 + values2

    @staticmethod
    def minus(values1: np.ndarray, values2: np.ndarray) -> np.ndarray:
        return values1 - values2

    @staticmethod
    def multiply(values1: np.ndarray, values2: np.ndarray) -> np.ndarray:
        return values1 * values2

    @staticmethod
    def divide(values1: np.ndarray, values2: np.ndarray) -> np.ndarray:
        return values1 / values2

    @staticmethod
    def negate(values: np.ndarray) -> np.ndarray:
        return values * (-1)

    @staticmethod
    def tabulate(systems: np.ndarray) -> np.ndarray:
        keys = set()
        for system in systems:
            assert isinstance(system, Mapping), type(system)
            keys.update(system.keys())
        keys = sorted(keys)
        table = []
        for system in systems:
            row = []
            for key in keys:
                # do not change to *if key in system*
                try:
                    row.append(system[key])
                except KeyError:
                    row.append(0)
            table.append(row)
        return np.asarray(table)

    @staticmethod
    def hstack(table: np.ndarray) -> np.ndarray:
        return np.hstack(table.transpose((1,0,2)))

    @staticmethod
    def getPrincipalComponents(dimensionality: int, data: np.ndarray) -> np.ndarray:
        principalComponents = PCA(n_components=dimensionality).fit_transform(data)
        assert principalComponents.shape[0] == data.shape[0]
        return principalComponents

    @staticmethod
    def convexHullHeight(space: np.ndarray) -> np.ndarray:
        return copy(ConvexHull(space).height)

    @staticmethod
    def simpleHeight(space: np.ndarray) -> np.ndarray:
        nRows, nColumns = space.shape
        if nColumns > 1:
            values = np.empty(nRows, dtype=float)
            for row in space:
                arg = np.empty((nColumns - 1, nRows), dtype=float)
                *arg[:], value = (space - row).T
                where = np.isclose(arg.all(axis=0), 0).nonzero()
                value = value[where]
                values[where] = value - value.min()
        else:
            values = (space - space.min(axis=0)).flatten()
        return values

    @staticmethod
    def getRelativeCHSpace(arguments: np.ndarray, properties: np.ndarray) -> np.ndarray:
        assert arguments.shape[0] == properties.shape[0]
        arguments = arguments.astype(float)
        totalBlocks = arguments.sum(axis=1)
        arguments /= totalBlocks.reshape((-1, 1))
        properties = np.nan_to_num(properties) / totalBlocks
        arguments[:,-1] = properties
        return arguments

    @staticmethod
    def getAbsoluteCHSpace(arguments: np.ndarray, properties: np.ndarray) -> np.ndarray:
        poolSize, argNum = arguments.shape
        assert properties.shape == (poolSize,)
        space = np.empty((poolSize, argNum + 1), dtype=float)
        space[:, :-1] = arguments
        space[:, -1] = np.nan_to_num(properties)
        return space

    @staticmethod
    def pareto(*arguments) -> np.ndarray:
        arguments = np.nan_to_num(np.asarray(arguments)).T.tolist()
        values = np.empty((len(arguments),), dtype=int)
        for i, front in enumerate(paretoRanking(arguments)):
            for ind in front:
                values[ind] = i
        return values

    @staticmethod
    def sort(population: list, allFitnesses: dict):
        """
        Method for sorting our population by fitness.

        :type fitness: list[tuple[str]]
        :param fitness: list of tuples ('property', 'direction'), where direction is 'min' or 'max'.
        :type population: list
        :param population: unsorted list of structures.
        :rtype: list
        :return: sorted population.
        """
        uniqueValues, ranking = np.unique([allFitnesses[s['ID']] for s in population], return_inverse=True)
        return [[population[ind] for ind in (ranking == rank).nonzero()[0]] for rank in range(len(uniqueValues))]
