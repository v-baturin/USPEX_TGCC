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

from .Antiseeds import Antiseeds

class Autofrac(object):
    '''

    '''

    def __init__(self, fractions : Dict[str, tuple], weightsLast, best : list, newFoundSystems : list, varOperators : list):
        '''

        :param population:
        :param best:
        :param newFoundSystems:
        :param varOperators:
        '''

        self.weightsLast = copy(weightsLast)
        self.weightsBest = Counter()
        for system in best:
            if system['howCome'] != 'Seeds' and system in newFoundSystems:
                self.weightsBest[system['howCome']] += 1

        self.initWeights = {}
        self.minFracs = {}
        self.maxFracs = {}
        for VO in varOperators:
            name = type(VO).__name__
            nl = name[0].lower() + name[1:]
            self.minFracs[name], self.maxFracs[name], self.initWeights[name] = fractions[nl] if nl in fractions else (0.0, 0.0, 0.0)

    def howMany(self, howCome, leftPopSize : int, totalPopSize : int):
        '''

        :param howCome:
        :param leftPopSize:
        :return:
        '''

        if self.weightsLast[howCome] == 0:
            initialNorm = sum(self.initWeights.values())
            frac = self.initWeights[howCome] / initialNorm if initialNorm > 0 else 0
            howMany = np.floor(frac * leftPopSize)
        else:
            lastNorm = np.fromiter((value for value in self.weightsLast.values()), dtype=int).sum()
            lastFrac = self.weightsLast[howCome] / lastNorm if lastNorm != 0 else 0
            weightsNorm = 0
            for key in set.union(set(self.weightsLast.keys()), set(self.weightsBest.keys())):
                weightsNorm += self.weightsBest[key] ** 2 / self.weightsLast[key] if self.weightsLast[key] != 0 else 0
            weight = self.weightsBest[howCome] ** 2 / self.weightsLast[howCome]
            frac = (lastFrac + weight / weightsNorm) / 2 if weightsNorm != 0 else lastFrac/2
            howMany = np.floor(frac * leftPopSize)
            howManyMin = np.floor(self.minFracs[howCome] * totalPopSize)
            howManyMax = np.floor(self.maxFracs[howCome] * totalPopSize)
            howMany = max(howManyMin, howMany)
            howMany = min(howManyMax, howMany)

        del self.weightsLast[howCome]
        del self.weightsBest[howCome]
        del self.initWeights[howCome]
        del self.minFracs[howCome]
        del self.maxFracs[howCome]
        return int(howMany)

'''
FunctionFolder/USPEX/3**/EA_3**.m
'''

class USPEXClassic(object):
    '''

    '''

    def __init__(self, pool, target, fingerprintUtility, optType, popSize : int, fractions : Dict[str, tuple],
                 initialPopSize=None, bestFrac:float=0.7, howManyDiverse=None, diversityTolerance = 0.5, debug = False, **kwargs):
        '''
        :param target: reference to configuration space object
        :param params: dictionary contains following parameters:
        popSize : int - size of population
        '''
        self.pool = pool
        self.target = target
        self.target.utilities.antiseeds = Antiseeds()
        self.fingerprintUtility = fingerprintUtility
        self.optType = optType
        self.fractions = fractions

        self.popSize = popSize

        if initialPopSize is not None:
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

    def __call__(self):
        '''
        :param oldPopulation: generation of new
        :param best:
        :param tournament:
        :param popSize:
        :return:
        '''

        if self.pool.generations:
            population = self.pool.generations[-1]['allSystems']
            newStructures = self.pool.generations[-1]['newSystems']
            fitness = self.pool.generations[-1]['fitness']
            fronts = fitness.sort(population + self._mostDiverse, fitness.getAllFitnesses(self.optType))
            sortedPopulation = list(chain.from_iterable(fronts))

            howManyProliferate = int(np.ceil(self.bestFrac * len(sortedPopulation)))
            best = sortedPopulation[:howManyProliferate]
            tournament = [(i + 1.0) ** 2 for i in reversed(range(howManyProliferate))]
            tournament /= np.sum(tournament)

            self._mostDiverse = self.determineMostDiverse(best, self.howManyDiverse, self.diversityTolerance)

            popSize = self.popSize
        else:
            newStructures, best, tournament, popSize = [], [], [], self.initialPopSize

        autofrac = Autofrac(self.fractions, self.weightsLast, best, newStructures, self.target.variationOperators)

        population = []
        actualParents = []

        for mutation in self.target.mutations:
            howCome = type(mutation).__name__
            howMany = autofrac.howMany(howCome, popSize - len(population), popSize)
            howMany = 0 if howMany < 0 else howMany
            if best:
                self.weightsLast[howCome] = howMany
                if hasattr(mutation, 'prepare'):
                    mutation.prepare()
                possibleParents = np.random.choice(best, size=10*howMany, replace=True, p=tournament)
                for parent in possibleParents:
                    if howMany <= 0:
                        break
                    try:
                        logger.debug(f"Trying {parent['ID']} parent.")
                        offsprings = mutation(parent)
                        for offspring in offsprings:
                            self.pool.assignID(offspring)
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

        for hybridization in self.target.hybridizations:
            howCome = type(hybridization).__name__
            howMany = autofrac.howMany(howCome, popSize - len(population), popSize)
            howMany = 0 if howMany < 0 else howMany
            if best:
                self.weightsLast[howCome] = howMany
                if hasattr(hybridization, 'prepare'):
                    hybridization.prepare()
                pairs = zip(np.random.choice(best, size=10 * howMany, replace=True, p=tournament),
                            np.random.choice(best, size=10 * howMany, replace=True, p=tournament))
                for parent1, parent2 in pairs:
                    if parent1['ID'] == parent2['ID']:
                        continue
                    if howMany <= 0:
                        break
                    try:
                        logger.debug(f"Trying {parent1['ID']} {parent2['ID']} parents.")
                        offsprings = hybridization(parent1,parent2)
                        for offspring in offsprings:
                            self.pool.assignID(offspring)
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

        for creation in self.target.creations:
            howCome = type(creation).__name__
            howMany = autofrac.howMany(howCome, popSize - len(population), popSize)
            howMany = 0 if howMany < 0 else howMany
            self.weightsLast[howCome] = howMany
            if hasattr(creation, 'prepare'):
                creation.prepare()
            for i in range(2 * howMany):
                if howMany <= 0:
                    break
                try:
                    offsprings = creation()
                    for offspring in offsprings:
                        self.pool.assignID(offspring)
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

        self.target.utilities.antiseeds.payPenalties(actualParents, self.pool.uniqueSystems, self.fingerprintUtility)

        if self.target.seeds is not None:
            seeds = self.target.seeds()
            for seed in seeds:
                self.pool.assignID(seed)
                seed['howCome'] = 'Seeds'
                seed['parent'] = "None"
                logger.info(f"Structure {seed['ID']} created from seed {seed['filename']}.")
            population.extend(seeds)

        return population

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
                return mostDiverse
            deltaTol /= 2
        raise RuntimeError(f"Can't clusterize population into {howManyDiverse} fractions.")

    def getMostDiverse(self) -> list:
        return copy(self._mostDiverse)