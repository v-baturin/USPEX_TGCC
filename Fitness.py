"""
USPEX.Common.Fitness
==================================

Data type representing rules of how we determine which systems are better.

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import sympy as smp
from copy import copy
from typing import List, Tuple
from collections.abc import Mapping
from itertools import combinations, chain
from sklearn.decomposition import PCA

from .ConvexHull import ConvexHull
from .paretoRanking import paretoRanking
from .Presets import presetFitness


class Fitness(object):

    ANTISEEDS_MAX = 0.005
    ANTISEEDS_SIGMA = 0.001

    def __init__(self, pool, utilities, fingerprintUtility):
        self.pool = pool
        self._poolHash = hash(self.pool)
        self.utilities = utilities
        self.fingerprintUtility = fingerprintUtility
        self._antiseedsCorrections = {}
        self._storedFitnesses = {}

    def __copy__(self):
        other = Fitness.__new__(Fitness)
        other.pool = self.pool
        other._poolHash = self._poolHash
        other.utilities = self.utilities
        other._antiseedsCorrections = copy(self._antiseedsCorrections)
        other._storedFitnesses = copy(self._storedFitnesses)
        return other

    @property
    def storedFitnesses(self):
        if self._poolHash != hash(self.pool):
            self._storedFitnesses = {}
            self._poolHash = hash(self.pool)
        return self._storedFitnesses

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

    def calcFitness(self, fitness):
        if fitness in presetFitness:
            fitness = presetFitness[fitness]
        if fitness not in self.storedFitnesses:
            if isinstance(fitness, tuple):
                funcName, *funcParams = fitness
                if not isinstance(funcName, str):
                    raise RuntimeError(f'Incorrect type {type(funcName)} of function {funcName}.')
                else:
                    funcName = funcName.split('.')
                    arguments = [self.calcFitness(param) for param in funcParams]
                    if len(funcName) == 1:
                        funcName, = funcName
                        self.storedFitnesses[fitness] = getattr(self, funcName)(*arguments)
                    elif len(funcName) == 2:
                        utility, funcName = funcName
                        self.storedFitnesses[fitness] = getattr(getattr(self.utilities, utility), funcName)(*arguments)
                    else:
                        raise RuntimeError(f"Too complex fitness {'.'.join(fitness)}.")
            elif isinstance(fitness, str):
                fitness = fitness.split('.')
                if len(fitness) == 1:
                    fitness, = fitness
                    value = [x[fitness] for x in self.pool.uniqueSystems]
                elif len(fitness) == 2:
                    utility, fitness = fitness
                    value = [getattr(getattr(self.utilities, utility), fitness)(x) for x in self.pool.uniqueSystems]
                else:
                    raise RuntimeError(f"Too complex fitness {'.'.join(fitness)}.")
                # unfortunately simple np.asarray spoils dictionaries
                if value and isinstance(value[0], Mapping):
                    valueArray = np.empty((len(value,)), dtype=type(value[0]))
                    for i, x in enumerate(value):
                        valueArray[i] = x
                else:
                    valueArray = np.asarray(value)
                return valueArray
            else:
                # just a parameter. return it without doing anything.
                return fitness
        return self.storedFitnesses[fitness]

    def getAllFitnesses(self, fitness):
        return dict(zip([s['ID'] for s in self.pool.uniqueSystems], self.calcFitness(fitness)))

    def getFitnessByID(self, fitness, ID):
        system = self.pool.allSystems[ID]
        if 'originalID' in system:
            ID = system['originalID']
        if fitness in presetFitness:
            fitness = presetFitness[fitness]
        value = None
        if fitness in self.storedFitnesses:
            IDs = [system['ID'] for system in self.pool.uniqueSystems]
            if ID in IDs:
                value =  self.storedFitnesses[fitness][IDs.index(ID)]
        elif isinstance(fitness, str):
            system = self.pool.allSystems[ID]
            fitness = fitness.split('.')
            if len(fitness) == 1:
                fitness, = fitness
                if fitness in system:
                    value = system[fitness]
            elif len(fitness) == 2:
                utility, fitness = fitness
                value = getattr(getattr(self.utilities, utility), fitness)(system)
            else:
                raise RuntimeError(f"Too complex fitness {'.'.join(fitness)}.")
        return value

    def payPenalties(self, population, pool):
        comb = list(combinations(population, 2))
        if comb:
            sigma = 0
            for s1, s2 in comb:
                sigma += self.fingerprintUtility.dist(s1, s2)
            sigma /= len(comb)
        else:
            sigma = 1
        sigma *= self.ANTISEEDS_SIGMA
        for system in pool:
            if system['ID'] in self._antiseedsCorrections:
                for ref_system in population:
                    dist = self.fingerprintUtility.dist(ref_system, system)
                    self._antiseedsCorrections[system['ID']] += np.exp(-dist**2/(2*sigma**2))
            else:
                self._antiseedsCorrections[system['ID']] = 0
                for ref_system in pool:
                    dist = self.fingerprintUtility.dist(ref_system, system)
                    self._antiseedsCorrections[system['ID']] += np.exp(-dist**2/(2*sigma**2))

    def aging(self, values: np.ndarray) -> np.ndarray:
        corrections = []
        for system in self.pool.uniqueSystems:
            if system['ID'] in self._antiseedsCorrections:
                corrections.append(self.ANTISEEDS_MAX * self._antiseedsCorrections[system['ID']])
            else:
                corrections.append(0)
        values += (values.mean() - values.min()) * np.asarray(corrections, dtype=float)
        return values

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
