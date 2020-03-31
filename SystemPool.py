'''
@file        Target.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        21 July 2016
@brief       Abstract class for different target configuration spaces
'''


import logging
logger = logging.getLogger(__name__)

from typing import List, Tuple

from .ParetoRanking import paretoRanking



def _fitnessRepresentation(fitness : List[Tuple[str, str]]):
    return '_'.join(f'{attr}_{direction}' for (attr, direction) in sorted(fitness, key=lambda entry: entry[0]))

class SystemPool(object):


    DEFAULT_FITNESS = []

    def __init__(self, config):
        '''

        :param fitness:
        :param stopFitness:
        :param kwargs:
        '''
        self.config = config
        self.best = {}
        self.uniqueSystems = []
        self._newID = 0

    def update(self, population : list):
        self.best = {}

        logger.info('Updating target: list of unique systems.')
        uniqueIDs = [system.ID for system in self.uniqueSystems]
        for system in population:
            if system.ID not in uniqueIDs:
                logger.debug('add new system %d to list of unique systems' % system.ID)
                uniqueIDs.append(system.ID)
                self.uniqueSystems.append(system)

    def newFoundSystems(self, population : list):
        logger.info('Determine new systems.')
        uniqueIDs = [system.ID for system in self.uniqueSystems]
        newFoundSystems = []
        for system in population:
            if system.ID not in uniqueIDs:
                logger.debug('found new system %d' % system.ID)
                newFoundSystems.append(system)
                uniqueIDs.append(system.ID)
        return newFoundSystems

    def cleanDuplicates(self, population : list):
        pass

    def setBest(self, fitness : List[Tuple[str, str]], best):
        self.best[_fitnessRepresentation(fitness)] = best

    def getBest(self, fitness : List[Tuple[str, str]]):
        return self.best[_fitnessRepresentation(fitness)]

    def sort(self, fitness : List[Tuple[str, str]], population : list):
        '''
        Method for sorting our population by
        :param population: unsorted list of structures.
        :return:
        '''
        populationFitnesses = []

        for system in population:
            systemFitnesses = []
            for attribute, direction in fitness:
                if direction == 'min':
                    factor = 1
                elif direction == 'max':
                    factor = -1
                else:
                    factor = 1
                    logger.debug('Incorrect optimization deirection "{}" using default "min"'.format(direction))
                if hasattr(self, attribute):
                    value = factor * getattr(self, attribute)(system)
                elif hasattr(self.config, attribute):
                        value = factor * getattr(self.config, attribute)(system)
                elif hasattr(system, attribute):
                    value = factor * getattr(system, attribute)
                else:
                    value = 0
                    logger.debug('Neither target {} nor system {} has attribute {}'
                                  ' which is set as fitness.'.format(self.__name__, system.ID, attribute))
                systemFitnesses.append(value)
            populationFitnesses.append(systemFitnesses)

        logger.debug('Fitnesses of this population are: {}'.format(populationFitnesses))
        ranking = paretoRanking(populationFitnesses)
        return [[population[index] for index in front] for front in ranking]
        # uniqueFinesses, ranking = np.unique(populationFitnesses, return_inverse=True)
        # return [[population[ind] for ind in (ranking == rank).nonzero()[0]] for rank in range(len(uniqueFinesses))]

    def assignID(self, system):
        system.ID = self._newID
        self._newID += 1
        system.__hash__ = system.systemHash
        system.__copy__ = system.copy
