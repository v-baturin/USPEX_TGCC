'''
@file        USPEXClassic.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        January 2017
@brief       Class that describes configuration space of systems under consideration.
'''

import logging
logger = logging.getLogger(__name__)


import random
from itertools import combinations, chain
from copy import copy
from typing import List, Tuple, Dict
from collections import Counter

import numpy as np

from .Autofrac import Autofrac

'''
FunctionFolder/USPEX/3**/EA_3**.m
'''

class USPEXClassic(object):
    '''

    '''

    def __init__(self, fingerprintUtility, fitness : List[Tuple[str, str]], popSize : int, fractions : Dict[str, tuple],
                 initialPopSize=None, bestFrac:float=0.7, howManyDiverse=None, diversityTolerance = 0.5, debug = False, **kwargs):
        '''
        :param target: reference to configuration space object
        :param params: dictionary contains following parameters:
        popSize : int - size of population
        '''
        self.fingerprintUtility = fingerprintUtility
        self.fitness = fitness
        self.fractions = fractions

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
        self.weightsLast = Counter()
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    def __call__(self, target, fitness, population : list, newStructures : list):
        '''
        :param oldPopulation: generation of new
        :param best:
        :param tournament:
        :param popSize:
        :return:
        '''

        if population is None:
            autofrac = Autofrac(self.fractions, Counter(), best=[], newFoundSystems=[], varOperators=target.variationOperators)
            best, tournament, popSize = [], [], self.initialPopSize
        else:
            extendedPopulation = copy(population)
            extendedPopulation.extend(self._mostDiverse)
            allFitnesses = fitness.getAllFitnesses(self.fitness)
            sortedPopulation = list(chain.from_iterable(fitness.sort(extendedPopulation, allFitnesses)))

            howManyProliferate = int(self.bestFrac * len(sortedPopulation))
            best = sortedPopulation[:howManyProliferate]
            tournament = [(i + 1.0) ** 2 for i in reversed(range(howManyProliferate))]
            tournament /= np.sum(tournament)

            self._mostDiverse = self.determineMostDiverse(best, self.howManyDiverse, self.diversityTolerance)
            autofrac = Autofrac(self.fractions, self.weightsLast, best, newStructures, target.variationOperators)

            popSize = self.popSize

        population = []
        actualParents = []

        if best:
            for mutation in target.mutations:
                howCome = type(mutation).__name__
                if hasattr(mutation, 'prepare'):
                    mutation.prepare()
                howMany = autofrac.howMany(howCome, popSize - len(population), popSize)
                self.weightsLast[howCome] = howMany
                possibleParents = np.random.choice(best, size=2*howMany, replace=True, p=tournament)
                for parent in possibleParents:
                    if howMany <= 0:
                        break
                    try:
                        logger.debug(f"Trying {parent['ID']} parent.")
                        offsprings = mutation(parent)
                        for offspring in offsprings:
                            target.pool.assignID(offspring)
                            offspring['howCome'] = howCome
                            offspring['parent'] = f"{parent['ID']}"
                            logger.info(f"System {offspring['ID']} successfully created by {offspring['howCome']} operator "
                                        f"from {offspring['parent']} parent.")
                        population.extend(offsprings)
                        howMany -= len(offsprings)
                        actualParents.append(parent)
                    except RuntimeError as e:
                        logger.debug(e)
                    except Exception as e:
                        logger.error(e, exc_info=True)
                if hasattr(mutation, 'standby'):
                    mutation.standby()

            for hybridization in target.hybridizations:
                howCome = type(hybridization).__name__
                if hasattr(hybridization, 'prepare'):
                    hybridization.prepare()
                howMany = autofrac.howMany(howCome, popSize - len(population), popSize)
                self.weightsLast[howCome] = howMany
                pairs = zip(np.random.choice(best, size=2 * howMany, replace=True, p=tournament),
                            np.random.choice(best, size=2 * howMany, replace=True, p=tournament))
                for parent1, parent2 in pairs:
                    if parent1['ID'] == parent2['ID']:
                        continue
                    if howMany <= 0:
                        break
                    try:
                        logger.debug(f"Trying {parent1['ID']} {parent2['ID']} parents.")
                        offsprings = hybridization(parent1,parent2)
                        for offspring in offsprings:
                            target.pool.assignID(offspring)
                            offspring['howCome'] = howCome
                            offspring['parent'] = f"{parent1['ID']} {parent2['ID']}"
                            logger.info(f"System {offspring['ID']} successfully created by {offspring['howCome']} operator "
                                        f"from {offspring['parent']} parents.")
                        population.extend(offsprings)
                        howMany -= len(offsprings)
                        actualParents.extend([parent1, parent2])
                    except RuntimeError as e:
                        logger.debug(e)
                    except Exception as e:
                        logger.error(e, exc_info=True)
                if hasattr(hybridization, 'standby'):
                    hybridization.standby()

        for creation in target.creations:
            howCome = type(creation).__name__
            if hasattr(creation, 'prepare'):
                creation.prepare()
            howMany = autofrac.howMany(howCome, popSize - len(population), popSize)
            self.weightsLast[howCome] = howMany
            for i in range(2 * howMany):
                if howMany <= 0:
                    break
                try:
                    offsprings = creation()
                    for offspring in offsprings:
                        target.pool.assignID(offspring)
                        offspring['howCome'] = howCome
                        offspring['parent'] = "None"
                        logger.info(f"System {offspring['ID']} successfully created by {offspring['howCome']} operator.")
                    population.extend(offsprings)
                    howMany -= len(offsprings)
                except RuntimeError as e:
                    logger.debug(e)
                except Exception as e:
                    logger.error(e, exc_info=True)
            if hasattr(creation, 'standby'):
                creation.standby()

        fitness.payPenalties(actualParents, target.pool.uniqueSystems)

        if target.seeds is not None:
            seeds = target.seeds()
            for seed in seeds:
                target.pool.assignID(seed)
                seed['howCome'] = type(target.seeds).__name__
                seed['parent'] = "None"
                logger.info(f"Structure {seed['ID']} created from seed {seed['filename']}.")
            population.extend(seeds)

        return population, (autofrac.weightsLast, autofrac.weightsBest)

    def determineMostDiverse(self, population : list, howManyDiverse: int, tolerance: float):
        """
        Here we perform clusterization in terms of distances between systems, assuming such distance is defined.
        For example atomic structures defines cosine distance in space of fingerprints.
        Such clusterization is an algorithm of determining a given amount (*howManyDiverse*) of systems from *population*
        so that the distances between them are greater than some threshold and all other systems in *population*
        lie in their vicinity with regard to the same threshold.
        The threshold determined automatically so such clusterization would be possible.
        :type population: list
        :param population: List of structures to be clusterized.
        :type howManyDiverse: int
        :param howManyDiverse: Amount of resulting structures.
        :type tolerance: float
        :param tolerance: Starting point for the threshold.
        :return:
        """
        deltaTol = tolerance / 2
        assert deltaTol > 0.000001
        while deltaTol > 0.000001:
            mostDiverse = []
            for system in population:
                goodSystem = True
                for ref_system in mostDiverse:
                    if self.fingerprintUtility.dist(system, ref_system) < tolerance:
                        goodSystem = False
                        break
                if goodSystem:
                    mostDiverse.append(system)
            if len(mostDiverse) < howManyDiverse:
                tolerance -= deltaTol
            elif len(mostDiverse) > howManyDiverse:
                tolerance += deltaTol
            else:
                break
            deltaTol /= 2
        return mostDiverse
