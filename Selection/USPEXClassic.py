'''
@file        USPEXClassic.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        January 2017
@brief       Class that describes configuration space of systems under consideration.
'''


import random
from itertools import combinations, chain
from copy import copy
from typing import List, Tuple

import numpy as np

from USPEX.Common.Target import Target
from USPEX.Common.VarOperator import VOFailed

from .Autofrac import Autofrac

'''
FunctionFolder/USPEX/3**/EA_3**.m
'''

class USPEXClassic(object):
    '''

    '''

    def __init__(self, target : Target, popSize : int, initialPopSize=None, bestFrac:float=0.7,
                 howManyDiverse=None, diversityTolerance = 0.5, **kwargs):
        '''
        :param target: reference to configuration space object
        :param params: dictionary contains following parameters:
        popSize : int - size of population
        '''
        self.target = target

        self.popSize = popSize

        if initialPopSize is not None:
            assert isinstance(initialPopSize, int) and initialPopSize > 0
            self.initialPopSize = initialPopSize
        else:
            self.initialPopSize = popSize
        self.bestFrac = bestFrac
        self.howManyDiverse = howManyDiverse if howManyDiverse else np.round(0.15*self.popSize)
        self.diversityTolerance = diversityTolerance
        self._mostDiverse = []

    def __call__(self, population : list, newStructures : list, fitness : List[Tuple[str, str]]):
        '''
        :param oldPopulation: generation of new
        :param best:
        :param tournament:
        :param popSize:
        :return:
        '''

        if population is None:
            autofrac = Autofrac(population=[], best=[], newFoundSystems=[], varOperators=self.target.variationOperators)
            best, tournament, popSize = [], [], self.initialPopSize
        else:
            for VO in self.target.variationOperators:
                VO.tune(population)

            extendedPopulation = copy(population)
            extendedPopulation.extend(self._mostDiverse)
            sortedPopulation = list(chain.from_iterable(self.target.pool.sort(fitness, extendedPopulation)))

            howManyProliferate = int(self.bestFrac * len(sortedPopulation))
            best = sortedPopulation[:howManyProliferate]
            tournament = [(i + 1.0) ** 2 for i in reversed(range(howManyProliferate))]
            tournament /= np.sum(tournament)

            self._mostDiverse = self.target.systemType.determineMostDiverse(best, self.howManyDiverse, self.diversityTolerance)
            autofrac = Autofrac(population, best, newStructures, self.target.variationOperators)

            popSize = self.popSize

        population = []
        actualParents = []

        for mutation in self.target.mutations:
            mutation.prepare()
            howMany = autofrac.howMany(mutation, popSize - len(population))
            if best:
                possibleParents = np.random.choice(len(best), size=howMany, replace=True, p=tournament)
            else:
                possibleParents = np.empty(0)
            for i in possibleParents:
                parent = best[i]
                if howMany <= 0:
                    break
                try:
                    offsprings = mutation(parent)
                    population.extend(offsprings)
                    howMany -= len(offsprings)
                    actualParents.append(parent)
                except VOFailed:
                    pass
            mutation.standby()

        for hybridization in self.target.hybridizations:
            hybridization.prepare()
            howMany = autofrac.howMany(hybridization, popSize - len(population))
            parents_pool = [parents for parents in combinations(range(len(best)), 2)]
            double_tournament = [tournament[parents[0]] * tournament[parents[1]] for parents in parents_pool]
            double_tournament = np.array(double_tournament) / sum(double_tournament)
            if parents_pool:
                pairs = [(best[parents_pool[ind][0]], best[parents_pool[ind][1]])
                         for ind in np.random.choice(len(parents_pool), size=howMany, p=double_tournament)]
            else:
                pairs = []
            for parent1, parent2 in random.sample(pairs, len(pairs)):
                if howMany <= 0:
                    break
                try:
                    offsprings = hybridization(parent1,parent2)
                    population.extend(offsprings)
                    howMany -= len(offsprings)
                    actualParents.extend([parent1, parent2])
                except VOFailed:
                    pass
            hybridization.standby()

        for creation in self.target.creations:
            creation.prepare()
            howMany = autofrac.howMany(creation, popSize - len(population))
            while howMany > 0:
                try:
                    offsprings = creation()
                    population.extend(offsprings)
                except VOFailed:
                    offsprings = tuple()
                howMany -= len(offsprings)
            creation.standby()

        self.target.pool.payPenalties(actualParents)
        return population, (autofrac.weightsLast, autofrac.weightsBest)

