"""
USPEX.Generators.Evolution
==========================


.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
import numpy as np
from copy import copy
from collections import Counter

from ..Expressions.Antiseeds import Antiseeds



logger = logging.getLogger(__name__)


class Autofrac(object):

    def __init__(self, fractions : dict[str, tuple], weightsLast, best : list, newFoundSystems : list, varOperators : list):
        """

        :param population:
        :param best:
        :param newFoundSystems:
        :param varOperators:
        """

        self.weightsLast = copy(weightsLast)
        self.weightsBest = Counter()
        for system in best:
            if system['.howCome.origin'] != 'Seeds' and system.ID in newFoundSystems:
                self.weightsBest[system['.howCome.origin']] += 1

        self.initWeights = {}
        self.minFracs = {}
        self.maxFracs = {}
        for VO in varOperators:
            name = type(VO).__name__
            nl = name[0].lower() + name[1:]
            self.minFracs[name], self.maxFracs[name], self.initWeights[name] = fractions[nl] if nl in fractions else (0.0, 0.0, 0.0)

    def howMany(self, howCome, leftPopSize : int, totalPopSize : int):
        """

        :param howCome:
        :param leftPopSize:
        :return:
        """

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


class Evolution(object):

    Target = None

    @classmethod
    def setTarget(cls, targetType: type):
        cls.Target = targetType

    def __init__(self, target: dict, optType: str | tuple, popSize: int, fractions: dict[str, tuple],
                 initialPopSize: int = None, bestFrac: float=0.7, howManyDiverse: int = None,
                 diversityTolerance: float = 0.5, antiseeds: dict = None, globalParentsPool: bool = False, debug=False,
                 **kwargs):
        """
        :param target: reference to configuration space object
        :param params: dictionary contains following parameters:
        popSize : int - size of population
        """
        self.target = self.Target(**target)
        self.optType = optType
        self.fractions = fractions
        antiseeds = {} if antiseeds is None else antiseeds
        self.antiseeds = Antiseeds(**antiseeds)

        self.popSize = popSize

        if initialPopSize is not None:
            self.initialPopSize = initialPopSize
        else:
            self.initialPopSize = popSize
        self.bestFrac = bestFrac
        self.howManyProliferate = int(np.ceil(self.bestFrac * self.popSize))
        self.howManyDiverse = howManyDiverse if howManyDiverse else int(np.round(0.15*self.popSize))
        self.diversityTolerance = diversityTolerance
        self.globalParentsPool = globalParentsPool
        self._mostDiverse = []
        self.weightsLast = Counter()
        self._newIDs = []
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    def __call__(self, generation):
        """
        :param oldPopulation: generation of new
        :param best:
        :param tournament:
        :param popSize:
        :return:
        """

        if generation is not None:
            self.antiseeds.payPenalties(generation.uniquePopulation, generation.uniqueSystems, self.target.metric)
            population = generation.uniqueSystems if self.globalParentsPool else generation.uniquePopulation
            optType = generation.goodSystems.createExpression(self.optType)
            generation.goodSystems.evaluate(self.optType)

            if not self.globalParentsPool:
                population = copy(population)
                for entry in self._mostDiverse:
                    population.addEntry(entry)

            fronts = population.fronts(optType)
            parentsPool = []
            tournament = []
            for i, front in enumerate(fronts):
                parentsPool.extend(front)
                tournament.extend([(len(fronts) - i) ** 2] * len(front))
            parentsPool = parentsPool[:self.howManyProliferate]
            tournament = tournament[:self.howManyProliferate]
            tournament /= np.sum(tournament)
            popSize = self.popSize
        else:
            parentsPool, tournament, popSize, optType = [], [], self.initialPopSize, None

        if not self.globalParentsPool:
            self._mostDiverse = self.determineMostDiverse(parentsPool)

        for VO in self.target.variationOperators:
            if hasattr(VO, 'tune'):
                VO.tune(parentsPool, optType)
        autofrac = Autofrac(self.fractions, self.weightsLast, parentsPool, self._newIDs, self.target.variationOperators)

        actualParents = []
        self._newIDs = []
        offsprings = self.target.createPool()

        for mutation in self.target.mutations:
            howCome = type(mutation).__name__
            howMany = autofrac.howMany(howCome, popSize - len(offsprings.getIDs()), popSize)
            howMany = 0 if howMany < 0 else howMany
            if parentsPool:
                self.weightsLast[howCome] = howMany
                if hasattr(mutation, 'prepare'):
                    mutation.prepare()
                possibleParents = np.random.choice(parentsPool, size=10*howMany, replace=True, p=tournament)
                for parent in possibleParents:
                    if howMany <= 0:
                        break
                    try:
                        logger.debug(f"Trying {parent['ID']} parent.")
                        for offspring in mutation(parent, offsprings.flavourFactory):
                            offspring.setProperty('howCome', howCome)
                            offspring.setProperty('parent', f"{parent['ID']}")
                            offspring.setProperty('isBad', False)
                            ID = offsprings.newEntry(offspring)
                            self._newIDs.append(ID)
                            howMany -= 1
                        actualParents.append(parent)
                    except RuntimeError as e:
                        logger.debug(e, exc_info=True)
                    except Exception as e:
                        logger.error(e, exc_info=True)
                if hasattr(mutation, 'standby'):
                    mutation.standby()

        for hybridization in self.target.hybridizations:
            howCome = type(hybridization).__name__
            howMany = autofrac.howMany(howCome, popSize - len(offsprings.getIDs()), popSize)
            howMany = 0 if howMany < 0 else howMany
            if parentsPool:
                self.weightsLast[howCome] = howMany
                if hasattr(hybridization, 'prepare'):
                    hybridization.prepare()
                pairs = zip(np.random.choice(parentsPool, size=10 * howMany, replace=True, p=tournament),
                            np.random.choice(parentsPool, size=10 * howMany, replace=True, p=tournament))
                for parent1, parent2 in pairs:
                    if parent1['ID'] == parent2['ID']:
                        continue
                    if howMany <= 0:
                        break
                    try:
                        logger.debug(f"Trying {parent1['ID']} {parent2['ID']} parents.")
                        for offspring in hybridization(parent1, parent2, offsprings.flavourFactory):
                            offspring.setProperty('howCome', howCome)
                            offspring.setProperty('parent', f"{parent1['ID']} {parent2['ID']}")
                            offspring.setProperty('isBad', False)
                            ID = offsprings.newEntry(offspring)
                            self._newIDs.append(ID)
                            howMany -= 1
                        actualParents.extend([parent1, parent2])
                    except RuntimeError as e:
                        logger.debug(e, exc_info=True)
                    except Exception as e:
                        logger.error(e, exc_info=True)
                if hasattr(hybridization, 'standby'):
                    hybridization.standby()

        for creation in self.target.creations:
            howCome = type(creation).__name__
            howMany = autofrac.howMany(howCome, popSize - len(offsprings.getIDs()), popSize)
            howMany = 0 if howMany < 0 else howMany
            self.weightsLast[howCome] = howMany
            if hasattr(creation, 'prepare'):
                creation.prepare()
            for i in range(2 * howMany):
                if howMany <= 0:
                    break
                try:
                    for offspring in creation(offsprings.flavourFactory):
                        offspring.setProperty('howCome', howCome)
                        offspring.setProperty('parent', "None")
                        offspring.setProperty('isBad', False)
                        ID = offsprings.newEntry(offspring)
                        self._newIDs.append(ID)
                        howMany -= 1
                except RuntimeError as e:
                    logger.debug(e, exc_info=True)
                except Exception as e:
                    logger.error(e, exc_info=True)
            if hasattr(creation, 'standby'):
                creation.standby()

        if self.target.seeds is not None:
            seeds = self.target.seeds(offsprings.flavourFactory)
            for seed in seeds:
                seed.setProperty('howCome', 'Seeds')
                seed.setProperty('parent', "None")
                seed.setProperty('isBad', False)
                ID = offsprings.newEntry(seed)
                self._newIDs.append(ID)
                logger.info(f"Seed filename is {seed['.filename']}.")

        # if not self.antiseeds.legacy:
        #     self.antiseeds.payPenalties(actualParents, generation.uniqueSystems, self.target.metric)
        return offsprings

    def determineMostDiverse(self, population: list):
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
        tolerance = self.diversityTolerance
        deltaTol = tolerance / 2
        mostDiverse = []
        while deltaTol > 0.000001:
            for system in population:
                for ref_system in mostDiverse:
                    if self.target.metric.equal(system, ref_system, tolerance):
                        break
                else:
                    mostDiverse.append(system)
            if len(mostDiverse) < self.howManyDiverse:
                tolerance -= deltaTol
            elif len(mostDiverse) > self.howManyDiverse:
                tolerance += deltaTol
            else:
                return mostDiverse
            deltaTol /= 2
            mostDiverse = []
        logger.debug(f"Can't clusterize population into {self.howManyDiverse} fractions.")
        return mostDiverse

    def getMostDiverse(self) -> list:
        return copy(self._mostDiverse)