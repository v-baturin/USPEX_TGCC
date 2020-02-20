'''
@file        USPEX.py
@author:     Pavel Bushlanov
@copyright:  2018 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        21 July 2016
@brief       Main model class
'''


import logging
from .Worker import Worker
from .Target import Target
from .Selection import Selection


logger = logging.getLogger(__name__)


class GlobalOptimizer(Worker):
    '''
    Main purpose of this class is to generate new structures
    that will be next optimized and selected best of them to the output.
    '''

    def __init__(self, target : dict, selection : dict, output):
        '''
        target: {type, params} - name of target system and its parameters; obligatory
        selection: {type, params} - name of selection to launch and its parameters; obligatory
        '''

        self.output = output
        self.target = Target(**target)
        if 'fitness' in target:
            self.fitness = target['fitness']
        else:
            self.fitness = self.target.pool.DEFAULT_FITNESS

        self.selection = Selection(self.target, **selection)
        self.population = None
        # List of new found structure on this particular step
        self.newStructures = None

        self.output.run(targetConfig=self.target.config, selectionConfig=self.selection.config)

    def run(self, population : list = None):
        '''
        Here we generate new set of structures
        :return:
        '''

        if population is None and self.population is not None:
            return self.population

        self.population, *analysis = self.selection.createPopulation(population, self.newStructures, self.fitness)

        self.save()
        self.output.run(analysis=analysis)
        return self.population

    def update(self, population : list):
        '''
        Updates state of optimized structures and write current state of them into the output
        :param population:
        '''
        self.target.pool.cleanDuplicates(population)
        self.newStructures = self.target.pool.newFoundSystems(population)
        self.target.pool.update(self.newStructures)
        self.target.pool.setBest(self.fitness, self.target.pool.rankSort(self.fitness, self.target.pool.uniqueSystems)[0])
        self.output.run(pool=self.target.pool)
