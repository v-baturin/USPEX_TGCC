'''
@file        Target.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        21 July 2016
@brief       Abstract class for different target configuration spaces
'''


import logging
from copy import copy
from itertools import chain

logger = logging.getLogger(__name__)


class Target(object):

    shortname = None

    config = None
    varOperators = []
    hybridizations = []
    mutations = []
    creations = []
    DEFAULT_FITNESS = []

    def __init__(self, rankSort):
        '''

        :param fitness:
        :param stopFitness:
        :param kwargs:
        '''
        self.rankSort = rankSort
        self.best = []
        self.uniqueSystems = []

    def update(self, population : list):
        logger.info('Updating target: best systems.')
        extendedPopulation = copy(population)
        extendedPopulation.extend(self.best)
        best = self.rankSort(extendedPopulation)[0]
        if set(best) != set(self.best):
            self.best = best

        logger.info('Updating target: list of unique systems.')
        uniqueIDs = [system.ID for system in self.uniqueSystems]
        newFoundSystems = []
        for system in population:
            if system.ID not in uniqueIDs:
                logger.debug('add new system %d to list of unique systems' % system.ID)
                newFoundSystems.append(system)
                uniqueIDs.append(system.ID)
        self.uniqueSystems.extend(newFoundSystems)
        self.uniqueSystems = list(chain.from_iterable(self.rankSort(self.uniqueSystems)))
        return newFoundSystems


    @property
    def state(self) -> tuple:
        return (self.best, self.uniqueSystems)
