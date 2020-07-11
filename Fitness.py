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
from itertools import combinations
from sklearn.decomposition import PCA

from .ConvexHull import ConvexHull
from .paretoRanking import paretoRanking

DIMENSIONALITY = 7


class Fintness(object):

    ANTISEEDS_MAX = 0.005
    ANTISEEDS_SIGMA = 0.001

    def __init__(self):
        self._antiseedsCorrections = {}

    def sort(self, fitness: List[Tuple[str, str]], population: list, pool):
        """
        Method for sorting our population by fitness.

        :type fitness: list[tuple[str]]
        :param fitness: list of tuples ('property', 'direction'), where direction is 'min' or 'max'.
        :type population: list
        :param population: unsorted list of structures.
        :rtype: list
        :return: sorted population.
        """
        IDs = []
        for system in population:
            for i, ref_system in enumerate(pool):
                if system.ID == ref_system.ID:
                    IDs.append(i)
        IDs = np.asarray(IDs, dtype=int)

        fitnessValues = []
        for attribute, direction in fitness:
            if attribute == 'formationEnergy':
                enthalpies = []
                compositions = []
                for system in pool:
                    composition = system.composition
                    compositions.append(composition)
                    enthalpies.append(system.enthalpy/sum(composition.values()))
                numIons = self.tabulate(compositions)
                numBlocks, blocks = smp.Matrix(numIons).T.rref()
                numBlocks = np.asarray(numBlocks, dtype = float)[:len(blocks)].T
                totalBlocks = numBlocks.sum(axis = 1)
                numBlocks /= totalBlocks.reshape((-1,1))
                enthalpies_per_block = np.asarray(enthalpies, dtype = float) / totalBlocks
                logger.debug(f'numBlocks: {numBlocks[:,:-1]}, enthalpies_per_block: {enthalpies_per_block}')
                values = self.convexHullHeight(numBlocks[:,:-1], enthalpies_per_block)[IDs]
            else:
                values = []
                for system in population:
                    if hasattr(system, attribute):
                        values.append(getattr(system, attribute))
                    else:
                        logger.info(f'System {system.ID} does not have attribute {attribute}. '
                                    f'Setting fitness value to 0.')
                values = np.asarray(values, dtype=float)
            try:
                direction, tail = direction.split('_')
                if tail == 'antiseeds':
                    useAntiseeds = True
                else:
                    logger.info(f'Incorrect format of fitness: {tail} is unknown option.')
                    useAntiseeds = False
            except:
                useAntiseeds = False
            if direction == 'max':
                values *= -1
            elif direction != 'min':
                logger.info('Incorrect optimization direction "{}" using default "min"'.format(direction))
            if useAntiseeds:
                values = self.getAntiseedsCorrections([{'ID': ID, 'value': value} for ID, value in zip(IDs, values)])
            fitnessValues.append(values)
        fitnessValues = np.nan_to_num(np.asarray(fitnessValues, dtype=float)).T
        logger.debug('Fitnesses of this population are: {}'.format(fitnessValues))
        ranking = paretoRanking(fitnessValues.tolist())
        return [[population[index] for index in front] for front in ranking]
        # uniqueFinesses, ranking = np.unique(populationFitnesses, return_inverse=True)
        # return [[population[ind] for ind in (ranking == rank).nonzero()[0]] for rank in range(len(uniqueFinesses))]

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

    def getAntiseedsCorrections(self, systems: List[dict]) -> np.ndarray:
        data_pd = pd.DataFrame(systems)
        values = data_pd.value.to_numpy()
        corrections = []
        for ID in data_pd.ID.to_list():
            if ID in self._antiseedsCorrections:
                corrections.append(self.ANTISEEDS_MAX * self._antiseedsCorrections[ID])
            else:
                corrections.append(0)
        values += (values.mean() - values.min()) * np.asarray(corrections, dtype=float)
        return values

    @staticmethod
    def tabulate(systems: List[dict]) -> np.ndarray:
        return np.nan_to_num(pd.DataFrame(systems).to_numpy())

    @staticmethod
    def getPrincipalComponents(systems: List[dict]) -> np.ndarray:
        N = len(systems[0])
        data_pd = pd.DataFrame(systems)
        data_pd[data_pd.isna()] = data_pd.apply(lambda row: row.loc[row.isna()].apply(lambda x: -np.ones(N)))
        data_np = np.hstack(np.array(data_pd.to_numpy().T.tolist(), dtype=float))
        return PCA(n_components=DIMENSIONALITY - 1).fit_transform(data_np)

    @staticmethod
    def convexHullHeight(arguments: np.ndarray, properties: np.ndarray) -> np.ndarray:
        assert arguments.shape[0] == properties.shape[0]
        systems = [{'argument': args, 'property': prop} for args, prop in zip(arguments, properties)]
        convexHull = ConvexHull()
        convexHull.extend(systems)
        return copy(convexHull.height)
