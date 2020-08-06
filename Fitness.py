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
from itertools import combinations, chain
from sklearn.decomposition import PCA

from .ConvexHull import ConvexHull
from .paretoRanking import paretoRanking

DIMENSIONALITY = 7


class Fitness(object):

    ANTISEEDS_MAX = 0.005
    ANTISEEDS_SIGMA = 0.001

    def __init__(self, pool):
        self.pool = pool
        self._antiseedsCorrections = {}

    def sort(self, fitness: tuple, population: list):
        """
        Method for sorting our population by fitness.

        :type fitness: list[tuple[str]]
        :param fitness: list of tuples ('property', 'direction'), where direction is 'min' or 'max'.
        :type population: list
        :param population: unsorted list of structures.
        :rtype: list
        :return: sorted population.
        """

        pairs = list(zip(self.pool, self.calcFitness(fitness)))
        values = []
        for system in population:
            for ref_system, value in pairs:
                if system.ID == ref_system.ID:
                    values.append(value)
                    break
        assert len(values) == len(population)

        uniqueValues, ranking = np.unique(values, return_inverse=True)
        return [[population[ind] for ind in (ranking == rank).nonzero()[0]] for rank in range(len(uniqueValues))]

    def calcFitness(self, fitness):
        if isinstance(fitness, tuple):
            funcName, *funcParams = fitness
            if not isinstance(funcName, str):
                raise RuntimeError(f'Incorrect type {type(funcName)} of function {funcName}.')
            elif not hasattr(self, funcName):
                raise RuntimeError(f'Function {funcName} not found in {type(self)}.')
            arguments = [self.calcFitness(param) for param in funcParams]
            return getattr(self, funcName)(*arguments)
        elif isinstance(fitness, str):
            return np.asarray([getattr(x, fitness) for x in self.pool])
        else:
            raise RuntimeError(f'Incorrect type {type(fitness)} of fitness {fitness}.')

    def payPenalties(self, population, pool):
        comb = list(combinations(population, 2))
        if comb:
            sigma = 0
            for s1, s2 in comb:
                assert hasattr(s1, 'dist')
                sigma += s1.dist(s1,s2)
            sigma /= len(comb)
        else:
            sigma = 1
        sigma *= self.ANTISEEDS_SIGMA
        for system in pool:
            assert hasattr(system, 'dist')
            if system.ID in self._antiseedsCorrections:
                for ref_system in population:
                    dist = system.dist(ref_system, system)
                    self._antiseedsCorrections[system.ID] += np.exp(-dist**2/(2*sigma**2))
            else:
                self._antiseedsCorrections[system.ID] = 0
                for ref_system in pool:
                    dist = system.dist(ref_system, system)
                    self._antiseedsCorrections[system.ID] += np.exp(-dist**2/(2*sigma**2))

    def getAntiseedsCorrections(self, values: np.ndarray) -> np.ndarray:
        corrections = []
        for system in self.pool:
            if system.ID in self._antiseedsCorrections:
                corrections.append(self.ANTISEEDS_MAX * self._antiseedsCorrections[system.ID])
            else:
                corrections.append(0)
        values += (values.mean() - values.min()) * np.asarray(corrections, dtype=float)
        return values

    @staticmethod
    def negate(values: np.ndarray) -> np.ndarray:
        return values * (-1)

    @staticmethod
    def tabulate(systems: np.ndarray) -> np.ndarray:
        return np.nan_to_num(pd.DataFrame(list(systems)).to_numpy())

    @staticmethod
    def getPrincipalComponents(systems: np.ndarray) -> np.ndarray:
        poolSize, N = systems.shape
        data_pd = pd.DataFrame(systems)
        data_pd[data_pd.isna()] = data_pd.apply(lambda row: row.loc[row.isna()].apply(lambda x: -np.ones(N)))
        data_np = np.hstack(np.array(data_pd.to_numpy().T.tolist(), dtype=float))
        principalComponents = PCA(n_components=DIMENSIONALITY - 1).fit_transform(data_np)
        assert principalComponents.shape[0] == poolSize
        return principalComponents

    @staticmethod
    def convexHullHeight(space: np.ndarray) -> np.ndarray:
        arguments = space[:,:-1]
        properties = space[:,-1]
        logger.debug(f'ConvexHull arguments: {arguments}, properties: {properties}')
        systems = [{'argument': args, 'property': prop} for args, prop in zip(arguments, properties)]
        convexHull = ConvexHull()
        convexHull.extend(systems)
        assert arguments.shape[0] == convexHull.height.shape[0]
        return copy(convexHull.height)

    @staticmethod
    def getRelativeCHSpace(arguments: np.ndarray, properties: np.ndarray) -> np.ndarray:
        assert arguments.shape[0] == properties.shape[0]
        numBlocks, blocks = smp.Matrix(arguments).T.rref()
        numBlocks = np.asarray(numBlocks, dtype=float)[:len(blocks)].T
        totalBlocks = numBlocks.sum(axis=1)
        numBlocks /= totalBlocks.reshape((-1, 1))
        properties = np.nan_to_num(properties) / totalBlocks
        assert numBlocks.shape[0] == properties.shape[0]
        numBlocks[:,-1] = properties
        return numBlocks

    @staticmethod
    def pareto(*arguments) -> np.ndarray:
        arguments = np.nan_to_num(np.asarray(arguments)).T.tolist()
        values = np.empty((len(arguments),), dtype=int)
        for i, front in enumerate(paretoRanking(arguments)):
            for ind in front:
                values[ind] = i
        return values
